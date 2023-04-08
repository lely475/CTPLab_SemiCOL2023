import random
from glob import glob
from typing import List

import albumentations as A
import albumentations.augmentations.functional as F
import cv2
import numpy as np
from albumentations.augmentations import RandomBrightnessContrast
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

from utils.config_templates.train import TrainConfig


class RandomBrightnessContrastChannel(RandomBrightnessContrast):
    """
    Per channel random brightness and contrast variation
    """

    def apply(
        self,
        img,
        alpha_r=1,
        beta_r=0,
        alpha_g=1,
        beta_g=0,
        alpha_b=1,
        beta_b=0,
        **params,
    ):
        img[:, :, 0] = F.brightness_contrast_adjust(
            img[:, :, 0], alpha_r, beta_r, self.brightness_by_max
        )
        img[:, :, 1] = F.brightness_contrast_adjust(
            img[:, :, 1], alpha_g, beta_g, self.brightness_by_max
        )
        img[:, :, 2] = F.brightness_contrast_adjust(
            img[:, :, 2], alpha_b, beta_b, self.brightness_by_max
        )
        return img

    def get_params(self):
        return {
            "alpha_r": 1.0
            + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
            "beta_r": 0.0
            + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
            "alpha_g": 1.0
            + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
            "beta_g": 0.0
            + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
            "alpha_b": 1.0
            + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
            "beta_b": 0.0
            + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
        }


class MeanStdTransfer(ImageOnlyTransform):
    """
    Replace image mean and std by mean and std of a reference
    """

    def __init__(
        self,
        means: List[np.ndarray],
        stds: List[np.ndarray],
        always_apply: bool = False,
        p: float = 0.5,
    ):
        self.means = means
        self.stds = stds
        super().__init__(always_apply, p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        ref_idx = np.random.choice(np.arange(len(self.means)))
        new_channel_mean = self.means[ref_idx]
        new_channel_std = self.stds[ref_idx]
        img_new = (img - np.mean(img, axis=(0, 1))) * new_channel_std / np.std(
            img, axis=(0, 1)
        ) + new_channel_mean
        img_new[img_new > 255] = 255
        img_new[img_new < 0] = 0
        return img_new.astype("uint8")


class MeanTransfer(ImageOnlyTransform):
    """
    Replace image mean by mean of a reference
    """

    def __init__(
        self,
        means: List[np.ndarray],
        always_apply: bool = False,
        p: float = 0.5,
    ):
        self.means = means
        super().__init__(always_apply, p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        mean_image_idx = np.random.choice(np.arange(len(self.means)))
        new_channel_mean = self.means[mean_image_idx]
        img_new = (img - np.mean(img, axis=(0, 1))) + new_channel_mean
        img_new[img_new > 255] = 255
        img_new[img_new < 0] = 0
        return img_new.astype("uint8")


def get_img_stat_transform(reference_mean_std: str, p: float) -> A.Compose:
    """
    Get reference means and std for image statistics transfer
    """
    means = []
    stds = []
    for x in sorted(glob(f"{reference_mean_std}/*.npy")):
        if "mean" in x:
            means.append(np.load(x))
        else:
            stds.append(np.load(x))

    img_stat_aug = A.OneOf(
        [MeanStdTransfer(means, stds), MeanTransfer(means)],
        p=p,
    )
    return img_stat_aug


def get_train_transform(config: TrainConfig) -> A.Compose:
    p = config.data_augmentation.aug_probability
    zoom_factor = min(
        config.data_augmentation.zoom_factor,
        1.0 - config.model.input_size / config.experiment.patch_size,
    )
    train_transform = A.Compose(
        [
            A.Flip(p=p),
            A.Transpose(p=p),
            A.RandomRotate90(p=p),
            get_img_stat_transform(config.data_augmentation.img_stat_ref_images, p=p),
            RandomBrightnessContrastChannel(
                p=p,
                brightness_limit=config.data_augmentation.bright_contrast_variation,
                contrast_limit=config.data_augmentation.bright_contrast_variation,
            ),
            A.RandomScale(
                scale_limit=(-zoom_factor, zoom_factor),
                p=p,
            ),
            A.RandomCrop(
                height=config.model.input_size,
                width=config.model.input_size,
                always_apply=True,
            ),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2(),
        ],
    )
    return train_transform


def get_test_transform(input_size: int) -> A.Compose:
    test_transform = A.Compose(
        [
            A.CenterCrop(input_size, input_size, always_apply=True),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2(),
        ]
    )
    return test_transform


def geometric_test_time_augmentation(img: np.ndarray) -> List[np.ndarray]:
    """apply geometric test time augmentation with all 8 possible rotate and flip conigurations"""
    transformed = []
    for flip in [None, 1]:
        for rotate in [
            None,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ]:
            t_img = cv2.flip(img, flip) if flip is not None else img
            t_img = cv2.rotate(t_img, rotate) if rotate is not None else t_img
            transformed.append(t_img)
    return transformed
