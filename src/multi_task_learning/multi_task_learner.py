import os
from typing import Any, Sequence, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim.lr_scheduler import ExponentialLR
from torchsummary import summary
from torchvision.transforms import CenterCrop, Pad

from models.multi_task_unet import UNetMultiTask
from utils.config_templates.train import TrainConfig
from utils.utils import get_input_from_output_size


class MultiTaskLearner(pl.LightningModule):
    """
    PyTorch Lightning module for training, validating and infering Multi Task Learner
    * instantiates Multi Task Model
    * defines all train and validation steps
    * configure callbacks for checkpointing
    * configure optimizer
    """

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.config = config
        self.model = UNetMultiTask(**self.config.model.arch.__dict__)

        self.num_classes = self.config.model.arch.num_classes
        input_size = self.config.model.input_size
        summary(self.model, input_size=(3, input_size, input_size), device="cpu")

    def on_fit_start(self) -> None:
        rank_zero_info(self.config)

        self.batch_size = self.config.experiment.batch_size
        self.sgm_weight = self.config.loss.sgm_weight
        self.lr = self.config.optimizer.lr
        input_size = self.config.model.input_size

        pad_in_size, pad_out_size = get_input_from_output_size(input_size)
        self.pad = Pad(int((pad_in_size - pad_out_size) / 2), padding_mode="reflect")

        out_size = self.model(
            torch.zeros(2, 3, input_size, input_size, device=self.device)
        )[0].shape[-1]
        self.center_crop = CenterCrop(size=(out_size, out_size))

        if self.trainer.is_global_zero:
            os.makedirs(f"{self.config.paths.output_path}/ckpts", exist_ok=True)
            os.makedirs(f"{self.config.paths.output_path}/tb_log", exist_ok=True)

    def on_train_start(self) -> None:
        self.num_train_batches = (
            len(self.trainer.train_dataloader) / self.trainer.accumulate_grad_batches
        )

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> STEP_OUTPUT:
        # Combine segmentation and tum det samples
        batch = self._combine_batches(batch["sgm"], batch["tum_det"])
        tiles, y_sgm, y_tum_det = batch

        # predict, center crop gt mask to pred size
        y_sgm, y_tum_det = y_sgm.long(), y_tum_det.float().unsqueeze(1)
        logits_sgm, logits_tum_det = self.model(tiles)
        y_sgm = self.center_crop(y_sgm)

        # loss
        loss_sgm = F.cross_entropy(logits_sgm, y_sgm, ignore_index=-1)
        mask = y_tum_det != -1  # ignore sgm patches
        loss_tum_det = F.binary_cross_entropy_with_logits(
            logits_tum_det[mask], y_tum_det[mask]
        )
        loss = self.sgm_weight * loss_sgm + (1 - self.sgm_weight) * loss_tum_det

        return {
            "loss": loss,
            "loss_sgm": loss_sgm,
            "loss_tum_det": loss_tum_det,
            "logits_sgm": logits_sgm,
            "logits_tum_det": logits_tum_det,
            "y_sgm": y_sgm,
            "y_tum_det": y_tum_det,
        }

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> STEP_OUTPUT:
        # Get batches, pad
        tiles, y_sgm, y_tum_det = batch
        tiles = self.pad.forward(tiles)

        # predict
        y_sgm, y_tum_det = y_sgm.long(), y_tum_det.float().unsqueeze(1)

        # loss
        logits_sgm, logits_tum_det = self.model(tiles)
        loss_sgm = F.cross_entropy(logits_sgm, y_sgm, ignore_index=-1)
        loss_tum_det = F.binary_cross_entropy_with_logits(logits_tum_det, y_tum_det)
        if torch.any(torch.isnan(loss_sgm)):  # ignore sgm_loss for tum det samples
            loss = loss_tum_det
        else:
            loss = loss_sgm + loss_tum_det

        return {
            "loss": loss,
            "loss_sgm": loss_sgm,
            "loss_tum_det": loss_tum_det,
            "logits_sgm": logits_sgm,
            "logits_tum_det": logits_tum_det,
            "y_sgm": y_sgm,
            "y_tum_det": y_tum_det,
        }

    def configure_optimizers(self) -> Any:
        """Loads the model optimizer"""
        if self.config.optimizer.name.upper() == "SGD":
            opt = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.optimizer.lr,
                **self.config.optimizer.params,
            )
            if self.config.optimizer.scheduler is not None:
                scheduler = ExponentialLR(opt, **self.config.optimizer.scheduler_params)
                return [opt], [{"scheduler": scheduler, "interval": "epoch"}]
            return opt
        else:
            return ValueError("Optimizer not implemented. Please modify source code.")

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        callbacks = []

        if self.config.experiment.full_set:
            filename = "ckpt-epoch={epoch:02d}-train_loss_sgm={train_loss_sgm:.2f}"
        else:
            filename = "ckpt-epoch={epoch:02d}-val_loss_sgm={val_loss_sgm_epoch/dataloader_idx_0:.2f}"
        callbacks.append(
            ModelCheckpoint(
                dirpath=f"{self.config.paths.output_path}/ckpts",
                filename=filename,
                save_top_k=10,
                auto_insert_metric_name=False,
                monitor="train_loss_sgm"
                if self.config.experiment.full_set
                else "val_loss_sgm_epoch/dataloader_idx_0",
                mode="min",
            )
        )
        return callbacks

    def _combine_batches(self, sgm_batch: Any, tum_det_batch: Any) -> Any:
        tiles = torch.cat((sgm_batch[0], tum_det_batch[0]), 0)
        y_sgm = torch.cat((sgm_batch[1], tum_det_batch[1]), 0)
        y_tum_det = torch.cat((sgm_batch[2], tum_det_batch[2]), 0)
        batch = [tiles, y_sgm, y_tum_det]
        return batch
