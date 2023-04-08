from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from multi_task_learning.metrics import TissueSgmMetrics, TumorDetectionMetrics
from multi_task_learning.multi_task_learner import MultiTaskLearner
from utils.tb_plots import (
    plot_batch_gallery,
    plot_probability_hist,
    plot_sgm_cm,
    plot_tum_det_cm,
    visualize_f1_scores,
)
from utils.utils import pretty_print_confusion_matrix


class MetricsLogger(Callback):
    """
    Compute and log metrics to Tensorboard
    """

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: MultiTaskLearner
    ) -> None:
        pl_module.train_sgm_metrics = TissueSgmMetrics(
            num_classes=pl_module.num_classes,
            ignore_index=-1,
            class_weights=pl_module.config.experiment.sgm_class_weights,
        )
        pl_module.train_tum_det_metrics = TumorDetectionMetrics(ignore_index=-1)

        pl_module.val_sgm_metrics = [
            TissueSgmMetrics(
                num_classes=pl_module.num_classes,
                ignore_index=-1,
                class_weights=pl_module.config.experiment.sgm_class_weights,
            )
            for _ in range(2)
        ]
        pl_module.val_tum_det_metrics = [
            TumorDetectionMetrics(ignore_index=-1) for _ in range(2)
        ]

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: MultiTaskLearner,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:

        pred_sgm = torch.argmax(outputs["logits_sgm"], dim=1, keepdim=True).squeeze(
            dim=1
        )
        pl_module.train_sgm_metrics.update(outputs["y_sgm"], pred_sgm)

        pred_tum_det = torch.sigmoid(outputs["logits_tum_det"]).flatten()
        gt_tum_det = outputs["y_tum_det"].long().flatten()
        pl_module.train_tum_det_metrics.update(gt_tum_det, pred_tum_det)

        step_metrics = {
            "train_loss": outputs["loss"],
            "train_loss_sgm": outputs["loss_sgm"],
            "train_loss_tum_det": outputs["loss_tum_det"],
        }
        pl_module.log_dict(
            step_metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=pl_module.batch_size,
        )

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: MultiTaskLearner
    ) -> None:
        epoch_metrics = {
            "train_sgm_f1": pl_module.train_sgm_metrics.f1,
            "train_tum_det_auroc": pl_module.train_tum_det_metrics.auroc,
            "step": float(pl_module.current_epoch),
        }
        pl_module.log_dict(
            epoch_metrics, on_step=False, on_epoch=True, batch_size=pl_module.batch_size
        )
        pl_module.logger.experiment.add_figure(
            "Train tumor detection probability histogram",
            plot_probability_hist(
                pl_module.train_tum_det_metrics.prob_hist_tumor,
                pl_module.train_tum_det_metrics.prob_hist_no_tumor,
            ),
            global_step=pl_module.current_epoch,
        )
        pl_module.logger.experiment.add_figure(
            "Train sgm classwise f1",
            visualize_f1_scores(pl_module.train_sgm_metrics.classwise_f1),
            global_step=pl_module.current_epoch,
        )
        pl_module.logger.experiment.add_figure(
            "Train sgm confusion matrix",
            plot_sgm_cm(pl_module.train_sgm_metrics.confusion_matrix, norm="all"),
            global_step=pl_module.current_epoch,
        )
        pl_module.logger.experiment.add_figure(
            "Train sgm norm confusion matrix",
            plot_sgm_cm(pl_module.train_sgm_metrics.confusion_matrix, norm="true"),
            global_step=pl_module.current_epoch,
        )
        pl_module.logger.experiment.add_figure(
            "Train tumor detection confusion matrix",
            plot_tum_det_cm(
                pl_module.train_tum_det_metrics.confusion_matrix, norm="all"
            ),
            global_step=pl_module.current_epoch,
        )

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: MultiTaskLearner,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not pl_module.trainer.sanity_checking:
            pred_sgm = torch.argmax(outputs["logits_sgm"], dim=1, keepdim=True).squeeze(
                dim=1
            )
            pl_module.val_sgm_metrics[dataloader_idx].update(outputs["y_sgm"], pred_sgm)

            pred_tum_det = torch.sigmoid(outputs["logits_tum_det"]).flatten()
            gt_tum_det = outputs["y_tum_det"].long().flatten()
            pl_module.val_tum_det_metrics[dataloader_idx].update(
                gt_tum_det, pred_tum_det
            )

            step_metrics = {
                "val_loss": outputs["loss"],
                "val_loss_tum_det": outputs["loss_tum_det"],
            }
            if not torch.any(torch.isnan(outputs["loss_sgm"])):
                step_metrics["val_loss_sgm"] = outputs["loss_sgm"]

            pl_module.log_dict(
                step_metrics,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=pl_module.batch_size,
            )

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: MultiTaskLearner
    ) -> None:
        if not pl_module.trainer.sanity_checking:
            epoch_metrics = {
                "val_sgm_f1": pl_module.val_sgm_metrics[0].f1,
                "val_tum_det_auroc_sgm_set": pl_module.val_tum_det_metrics[0].auroc,
                "val_tum_det_auroc_tum_det_set": pl_module.val_tum_det_metrics[1].auroc,
                "val_loss_sgm_set": trainer.logged_metrics[
                    "val_loss_epoch/dataloader_idx_0"
                ],
                "step": float(pl_module.current_epoch),
            }
            pl_module.log_dict(
                epoch_metrics,
                on_step=False,
                on_epoch=True,
                batch_size=pl_module.batch_size,
            )

            pl_module.logger.experiment.add_figure(
                "Validation sgm classwise f1",
                visualize_f1_scores(pl_module.val_sgm_metrics[0].classwise_f1),
                global_step=pl_module.current_epoch,
            )
            pl_module.logger.experiment.add_figure(
                "Validation tumor detection probability histogram, dataloader: sgm",
                plot_probability_hist(
                    pl_module.val_tum_det_metrics[0].prob_hist_tumor,
                    pl_module.val_tum_det_metrics[0].prob_hist_no_tumor,
                ),
                global_step=pl_module.current_epoch,
            )
            pl_module.logger.experiment.add_figure(
                "Validation tumor detection probability histogram, dataloader: tum det",
                plot_probability_hist(
                    pl_module.val_tum_det_metrics[1].prob_hist_tumor,
                    pl_module.val_tum_det_metrics[1].prob_hist_no_tumor,
                ),
                global_step=pl_module.current_epoch,
            )
            pl_module.logger.experiment.add_figure(
                "Validation sgm norm confusion matrix",
                plot_sgm_cm(pl_module.val_sgm_metrics[0].confusion_matrix, norm="true"),
                global_step=pl_module.current_epoch,
            )
            pl_module.logger.experiment.add_figure(
                "Validation sgm confusion matrix",
                plot_sgm_cm(pl_module.val_sgm_metrics[0].confusion_matrix, norm="all"),
                global_step=pl_module.current_epoch,
            )
            pl_module.logger.experiment.add_figure(
                "Validation tumor detection confusion matrix sgm set",
                plot_tum_det_cm(
                    pl_module.val_tum_det_metrics[0].confusion_matrix, norm="all"
                ),
                global_step=pl_module.current_epoch,
            ),
            pl_module.logger.experiment.add_figure(
                "Validation tumor detection confusion matrix tum det set",
                plot_tum_det_cm(
                    pl_module.val_tum_det_metrics[1].confusion_matrix,
                    norm="all",
                ),
                global_step=pl_module.current_epoch,
            )

    def on_save_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: MultiTaskLearner,
        checkpoint: Dict[str, Any],
    ) -> None:
        checkpoint["val_sgm_metrics"] = pl_module.val_sgm_metrics
        checkpoint["train_sgm_metrics"] = pl_module.train_sgm_metrics
        checkpoint["val_tum_det_metrics"] = pl_module.val_tum_det_metrics
        checkpoint["train_tum_det_metrics"] = pl_module.train_tum_det_metrics

    def on_load_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: MultiTaskLearner,
        checkpoint: Dict[str, Any],
    ) -> None:
        pl_module.train_sgm_metrics = checkpoint["train_sgm_metrics"]
        pl_module.train_tum_det_metrics = checkpoint["train_tum_det_metrics"]
        pl_module.val_sgm_metrics = checkpoint["val_sgm_metrics"]
        pl_module.val_tum_det_metrics = checkpoint["val_tum_det_metrics"]


class TxtFileLogger(Callback):
    """
    Save metrics confusion matrix, f1 and AUROC to txt file
    """

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: MultiTaskLearner
    ) -> None:
        sgm_cm = pl_module.train_sgm_metrics.confusion_matrix
        sgm_f1 = pl_module.train_sgm_metrics.f1
        tum_det_cm = pl_module.train_tum_det_metrics.confusion_matrix
        tum_det_auroc = pl_module.train_tum_det_metrics.auroc

        with open(
            f"{pl_module.config.paths.output_path}/train_sgm_conf_matrix.txt", "a"
        ) as f:
            f.write(
                f"Epoch {pl_module.current_epoch}, total: {np.sum(sgm_cm)}, F1: {sgm_f1}:\n"
            )
            f.write(pretty_print_confusion_matrix(sgm_cm, list(range(1, 11))))
        with open(
            f"{pl_module.config.paths.output_path}/train_tum_det_conf_matrix.txt", "a"
        ) as f:
            f.write(
                f"Epoch {pl_module.current_epoch}, total: {np.sum(tum_det_cm)}, Auroc: {tum_det_auroc}:\n"
            )
            f.write(pretty_print_confusion_matrix(tum_det_cm, ["No Tumor", "Tumor"]))

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: MultiTaskLearner
    ) -> None:
        if not trainer.sanity_checking:
            sgm_cm = pl_module.val_sgm_metrics[0].confusion_matrix
            sgm_f1 = pl_module.val_sgm_metrics[0].f1
            tum_det_cm_sgm_set = pl_module.val_tum_det_metrics[0].confusion_matrix
            tum_det_cm_tum_det_set = pl_module.val_tum_det_metrics[1].confusion_matrix
            tum_det_auroc_sgm = pl_module.val_tum_det_metrics[0].auroc
            tum_det_auroc_tum_det = pl_module.val_tum_det_metrics[1].auroc
            with open(
                f"{pl_module.config.paths.output_path}/val_sgm_conf_matrix.txt", "a"
            ) as f:
                f.write(
                    f"Epoch {pl_module.current_epoch}, total: {np.sum(sgm_cm)}, F1: {sgm_f1}:\n"
                )
                f.write(pretty_print_confusion_matrix(sgm_cm, list(range(1, 11))))
            with open(
                f"{pl_module.config.paths.output_path}/val_tum_det_conf_matrix.txt", "a"
            ) as f:
                f.write(
                    f"Epoch {pl_module.current_epoch}, Sgm Set, "
                    f"total: {np.sum(tum_det_cm_sgm_set)}, Auroc: {tum_det_auroc_sgm}:\n"
                )
                f.write(
                    pretty_print_confusion_matrix(
                        tum_det_cm_sgm_set, ["No Tumor", "Tumor"]
                    )
                )
                f.write(
                    f"Epoch {pl_module.current_epoch}, Tum Det Set, "
                    f"total: {np.sum(tum_det_cm_tum_det_set)}, Auroc: {tum_det_auroc_tum_det}:\n"
                )
                f.write(
                    pretty_print_confusion_matrix(
                        tum_det_cm_tum_det_set, ["No Tumor", "Tumor"]
                    )
                )


class GalleryLogger(Callback):
    """
    Log gallery of last batch inputs, preds and gts to tensorboard
    """

    def __init__(self) -> None:
        super().__init__()
        self._once = [True, True]

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: MultiTaskLearner,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if batch_idx == pl_module.num_train_batches - 1:
            if pl_module.current_epoch == 0:
                batch = pl_module._combine_batches(batch["sgm"], batch["tum_det"])
                tiles, y_sgm, y_tum_det = batch
                if pl_module.batch_size > 16:
                    idx = list(range(8)) + list(
                        range(pl_module.batch_size - 8, pl_module.batch_size)
                    )
                    tiles, y_sgm, y_tum_det = tiles[idx], y_sgm[idx], y_tum_det[idx]
                    pl_module.track_progress_batch = [tiles, y_sgm, y_tum_det]
            else:
                tiles, y_sgm, y_tum_det = pl_module.track_progress_batch
                if not tiles.is_cuda:
                    tiles, y_sgm, y_tum_det = (
                        tiles.to("cuda"),
                        y_sgm.to("cuda"),
                        y_tum_det.to("cuda"),
                    )

            padded_tiles = pl_module.pad.forward(tiles)
            logits_sgm, logits_tum_det = pl_module.model(padded_tiles)

            pred_sgm = torch.argmax(logits_sgm, dim=1, keepdim=True)
            pred_tum_det = torch.sigmoid(logits_tum_det).flatten()
            if pl_module.current_epoch == 0:
                pl_module.logger.experiment.add_figure(
                    "Ground truth",
                    plot_batch_gallery(tiles, y_sgm, y_tum_det),
                    global_step=pl_module.current_epoch,
                )
            pl_module.logger.experiment.add_figure(
                "Prediction",
                plot_batch_gallery(tiles, pred_sgm, pred_tum_det),
                global_step=pl_module.current_epoch,
            )

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: MultiTaskLearner,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not trainer.sanity_checking:
            if batch_idx == 25:
                tiles, y_sgm, y_tum_det = batch
                if pl_module.batch_size > 16:
                    tiles, y_sgm, y_tum_det = tiles[:16], y_sgm[:16], y_tum_det[:16]

                padded_tiles = pl_module.pad.forward(tiles)
                logits_sgm, logits_tum_det = pl_module.model(padded_tiles)
                pred_sgm = torch.argmax(logits_sgm, dim=1, keepdim=True)
                pred_tum_det = torch.where(
                    torch.sigmoid(logits_tum_det) > 0.5, 1, 0
                ).flatten()

                if self._once[dataloader_idx]:
                    pl_module.logger.experiment.add_figure(
                        f"Val Ground truth {'sgm batch' if dataloader_idx == 0 else 'tum det batch'}",
                        plot_batch_gallery(tiles, y_sgm, y_tum_det, num_cols=8),
                        global_step=pl_module.current_epoch,
                    )
                    self._once[dataloader_idx] = False

                pl_module.logger.experiment.add_figure(
                    f"Val Prediction {'sgm batch' if dataloader_idx == 0 else 'tum det batch'}",
                    plot_batch_gallery(tiles, pred_sgm, pred_tum_det, num_cols=8),
                    global_step=pl_module.current_epoch,
                )

    def on_save_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: MultiTaskLearner,
        checkpoint: Dict[str, Any],
    ) -> None:
        checkpoint["track_progress_batch"] = pl_module.track_progress_batch

    def on_load_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: MultiTaskLearner,
        checkpoint: Dict[str, Any],
    ) -> None:
        pl_module.track_progress_batch = checkpoint["track_progress_batch"]
