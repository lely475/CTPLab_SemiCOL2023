from typing import Any, Dict, List, Tuple, Union

import albumentations as A
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    WeightedRandomSampler,
    Subset,
)
from utils.config_templates.train import TrainConfig
from utils.constants import NUM_SGM_CLASSES, TEST_SEED
from dataloader.data_augmentation import get_test_transform, get_train_transform


class MultiTaskDataModule(pl.LightningDataModule):
    """Creates training and validation sets for tissue segmentation and tumor detection"""

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        self.batch_size = self.config.experiment.batch_size
        self.sgm_ratio = self.config.experiment.sgm_ratio

        self.train_rng = torch.Generator()
        self.train_rng.manual_seed(self.config.experiment.seed)
        self.test_rng = torch.Generator()
        self.test_rng.manual_seed(TEST_SEED)
        self.test_rng_np = np.random.default_rng(TEST_SEED)
        self.train_transform = get_train_transform(self.config)
        self.test_transform = get_test_transform(self.config.model.input_size)

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            self.train_sgm_csv = self.config.paths.train_sgm_csv
            self.val_sgm_csv = self.config.paths.val_sgm_csv
            self.train_tum_det_csv = self.config.paths.train_tum_det_csv
            self.val_tum_det_csv = self.config.paths.val_tum_det_csv
            self.batch_size = self.config.experiment.batch_size
            self.num_workers = self.config.experiment.num_workers
            self.sgm_ratio = self.config.experiment.sgm_ratio

    def train_dataloader(self):
        if self.config.experiment.full_set:
            sgm_csv = [self.config.paths.train_sgm_csv, self.config.paths.val_sgm_csv]
            tum_det_csv = [
                self.config.paths.train_tum_det_csv,
                self.config.paths.val_tum_det_csv,
            ]
        else:
            sgm_csv = self.config.paths.train_sgm_csv
            tum_det_csv = self.config.paths.train_tum_det_csv

        sgm_ds = TissueSegmentationDataset(sgm_csv, self.train_transform)
        tum_det_ds = TumorDetectionDataset(tum_det_csv, self.train_transform)
        debug = len(sgm_ds)
        sgm_loader = DataLoader(
            sgm_ds,
            sampler=WeightedRandomSampler(
                self.get_sgm_weights(),
                num_samples=debug,
                replacement=True,
                generator=self.train_rng,
            ),
            batch_size=int(self.batch_size * self.sgm_ratio),
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            generator=self.train_rng,
            pin_memory=True,
            persistent_workers=True,
        )
        tum_det_loader = DataLoader(
            tum_det_ds,
            batch_size=int(self.batch_size * (1 - self.sgm_ratio)),
            sampler=RandomSampler(
                tum_det_ds, num_samples=debug, generator=self.train_rng
            ),
            num_workers=self.num_workers,
            drop_last=True,
            generator=self.train_rng,
            pin_memory=True,
            persistent_workers=True,
        )
        loaders = {"sgm": sgm_loader, "tum_det": tum_det_loader}
        combined_loader = CombinedLoader(loaders, mode="min_size")
        return combined_loader

    def val_dataloader(self):
        sgm_ds = TissueSegmentationDataset(
            self.config.paths.val_sgm_csv, self.test_transform
        )
        tum_det_ds = TumorDetectionDataset(
            self.config.paths.val_tum_det_csv, self.test_transform
        )
        sgm_ds_idxs = self.test_rng_np.choice(
            np.arange(len(sgm_ds)), size=len(sgm_ds), replace=False
        )
        tum_det_idxs = self.test_rng_np.choice(
            len(tum_det_ds), size=len(sgm_ds), replace=False
        )
        shuffled_sgm_ds = Subset(sgm_ds, sgm_ds_idxs)
        reduced_tum_det_ds = Subset(tum_det_ds, tum_det_idxs)

        sgm_loader = DataLoader(
            shuffled_sgm_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            generator=self.test_rng,
            persistent_workers=True,
        )
        tum_det_loader = DataLoader(
            reduced_tum_det_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            generator=self.test_rng,
            persistent_workers=True,
        )

        return [sgm_loader, tum_det_loader]

    def get_sgm_weights(self) -> List[float]:
        sgm_weights = compute_sample_weights(
            self.config.paths.train_sgm_csv, self.config.experiment.patch_size
        )
        return sgm_weights


class TissueSegmentationDataset(Dataset):
    """Dataset for multi-class segmentation task"""

    def __init__(
        self,
        csv_file: Union[str, List[str]],
        transform: A.Compose,
    ):
        if type(csv_file) == list:
            self.df = pd.read_csv(csv_file[0])
            for file in csv_file[1:]:
                new_df = pd.read_csv(file)
                self.df = pd.concat([self.df, new_df])
        else:
            self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        bgr_img = np.load(self.df.iloc[idx]["tile_path"])
        r_channel = bgr_img[:, :, 2]
        b_channel = bgr_img[:, :, 0]
        rgb_img = bgr_img.copy()

        # swap B and R channel
        rgb_img[:, :, 0] = r_channel
        rgb_img[:, :, 2] = b_channel

        mask = np.load(self.df.iloc[idx]["tile_mask_path"]).astype("int")
        augmented = self.transform(image=rgb_img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]
        masks = mask - 1
        label = 1 if 1 in mask else 0  # assign tum det label
        return img, masks, label


class TumorDetectionDataset(Dataset):
    """Dataset for binary classification task"""

    def __init__(self, csv_file: Union[str, List[str]], transform=A.Compose) -> None:
        if type(csv_file) == list:
            self.df = pd.read_csv(csv_file[0])
            for file in csv_file[1:]:
                new_df = pd.read_csv(file)
                self.df = pd.concat([self.df, new_df])
        else:
            self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        img_path, label = self.df.iloc[idx, :][["tile_path", "label"]]
        bgr_img = np.load(img_path)
        r_channel = bgr_img[:, :, 2]
        b_channel = bgr_img[:, :, 0]
        rgb_img = bgr_img.copy()

        # swap B and R channel
        rgb_img[:, :, 0] = r_channel
        rgb_img[:, :, 2] = b_channel
        if rgb_img.max() == 0:
            raise ValueError(f"Image {img_path} is all zeros")

        img = self.transform(image=rgb_img)["image"]
        masks = np.ones((img.shape[1], img.shape[2]), dtype="int") * -1
        return img, masks, label


def compute_sample_weights(csv_path: str, img_size: int = 300) -> List[float]:
    """Compute class weights based on the class distribution"""

    print("Computing sample weights...")
    df = pd.read_csv(csv_path)
    class_pixels = df[[str(label) for label in range(1, NUM_SGM_CLASSES)]].sum().values
    total_pixels = class_pixels.sum()
    class_weights = total_pixels / class_pixels
    sample_weights = df.apply(get_sample_weight, args=(class_weights, img_size), axis=1)
    print("Finished computing sample weights.")
    return sample_weights.to_list()


def get_sample_weight(
    row: Dict[Any, Any], class_weights: List[float], img_size: int = 300
):
    sample_weight = 0
    for label in range(1, NUM_SGM_CLASSES):
        if row[str(label)] > 0:
            sample_weight += row[str(label)] * class_weights[label - 1] / img_size**2
    return sample_weight
