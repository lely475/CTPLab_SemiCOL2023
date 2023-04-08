from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


class TissueSgmMetrics:
    """Metrics class for tissue segmentation"""

    def __init__(
        self,
        num_classes: int,
        ignore_index: Tuple[List[int], int],
        class_weights: List[float],
    ) -> None:
        self._count = 0
        self._num_classes = num_classes
        self._ignore_index = ignore_index
        self._confusion_matrix = torch.zeros(
            (self._num_classes, self._num_classes), device="cuda"
        )
        self._class_weights = class_weights

    def remove_ignored_labels(
        self, gt: torch.Tensor, pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """remove all pixels that belong to ignored class"""
        ignore_index = (
            [self._ignore_index]
            if type(self._ignore_index) != list
            else self._ignore_index
        )
        mask = torch.full(size=gt.shape, fill_value=True, dtype=bool, device=gt.device)
        for label in ignore_index:
            mask = mask & (gt != label) & (pred != label)
        gt = gt[mask]
        pred = pred[mask]
        return gt, pred

    def update(self, gt: torch.Tensor, pred: torch.Tensor) -> None:
        """update confusion matrix"""
        gt_mask, pred_mask = self.remove_ignored_labels(gt, pred)
        self._confusion_matrix += compute_confusion_matrix(
            gt_mask, pred_mask, self._num_classes
        )
        self._count += 1

    @property
    def confusion_matrix(self) -> np.ndarray:
        cm = self._confusion_matrix.detach().cpu().numpy()
        return cm

    @property
    def classwise_f1(self) -> np.ndarray:
        f1 = []
        for label in range(self._num_classes):
            tp = self.confusion_matrix[label, label]
            fp = np.sum(self.confusion_matrix[label, :]) - tp
            fn = np.sum(self.confusion_matrix[:, label]) - tp
            with np.errstate(divide="ignore", invalid="ignore"):
                class_f1 = tp / (tp + 0.5 * (fp + fn))
            f1.append(class_f1)
        return np.array(f1)

    @property
    def f1(self) -> np.ndarray:
        return np.sum(self.classwise_f1 * self._class_weights)

    @property
    def classwise_iou(self) -> np.ndarray:
        iou = []
        for label in range(self._num_classes):
            tp = self._confusion_matrix[label, label]
            fp = np.sum(self.confusion_matrix[label, :])
            fn = np.sum(self.confusion_matrix[:, label])
            with np.errstate(divide="ignore", invalid="ignore"):
                class_iou = tp / (tp + fp + fn)
            iou.append(class_iou)
        return np.array(iou)

    @property
    def iou(self) -> np.ndarray:
        return np.sum(self.classwise_f1 * self._class_weights)


class TumorDetectionMetrics:
    """Metrics class for tumor detection"""

    def __init__(self, ignore_index: int, num_bins: int = 100) -> None:
        self._count = 0
        self._ignore_index = ignore_index
        self._confusion_matrix = torch.zeros((2, 2), device="cuda")
        self._auroc = 0.0
        self._num_bins = num_bins
        self._pos_hist = torch.zeros(num_bins, device="cuda")
        self._neg_hist = torch.zeros(num_bins, device="cuda")

    def remove_ignored_labels(
        self, gt: torch.Tensor, pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """remove all pixels that belong to ignored class"""
        mask = gt != self._ignore_index
        gt = gt[mask]
        pred = pred[mask]
        return gt, pred

    def update(self, gt: torch.Tensor, pred: torch.Tensor) -> None:
        """update confusion matrix, histograms and auroc"""
        gt_mask, pred_mask = self.remove_ignored_labels(gt, pred)
        class_pred = torch.where(pred_mask > 0.5, 1, 0)
        self._confusion_matrix += compute_confusion_matrix(gt_mask, class_pred, 2)
        self._pos_hist += torch.histc(
            pred_mask[gt_mask == 1], bins=self._num_bins, min=0.0, max=1.0
        )
        self._neg_hist += torch.histc(
            pred_mask[gt_mask == 0], bins=self._num_bins, min=0.0, max=1.0
        )
        gt_np, pred_np = (
            gt_mask.detach().cpu().numpy(),
            pred_mask.detach().cpu().numpy(),
        )
        try:
            self._auroc += roc_auc_score(gt_np, pred_np)
        except (ValueError):
            raise ValueError(
                "Only one class present in y_true. ROC AUC score is not defined in that case. ",
                gt_np,
                pred_np,
            )
        self._count += 1

    @property
    def prob_hist_tumor(self) -> np.ndarray:
        return self._pos_hist.detach().cpu().numpy()

    @property
    def prob_hist_no_tumor(self) -> np.ndarray:
        return self._neg_hist.detach().cpu().numpy()

    @property
    def prob_hist(self) -> np.ndarray:
        combined_hist = (
            self._neg_hist.detach().cpu().numpy()
            + self._pos_hist.detach().cpu().numpy()
        )
        return combined_hist

    @property
    def confusion_matrix(self) -> np.ndarray:
        cm = self._confusion_matrix.detach().cpu().numpy()
        return cm

    @property
    def f1(self) -> np.ndarray:
        cm = self.confusion_matrix
        f1 = (
            cm[1, 1] / (cm[1, 1] + 0.5 * (cm[0, 1] + cm[1, 0])) if cm[1, 1] > 0 else 0.0
        )
        return f1

    @property
    def acc(self) -> np.ndarray:
        cm = self.confusion_matrix
        return np.trace(cm) / np.sum(cm)

    @property
    def balanced_acc(self) -> np.ndarray:
        cm = self._confusion_matrix
        return 0.5 * (self.recall + cm[0, 0] / (cm[0, 0] + cm[1, 0]))

    @property
    def precision(self) -> np.ndarray:
        cm = self.confusion_matrix
        precision = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if cm[1, 1] > 0 else 0.0
        return precision

    @property
    def recall(self) -> np.ndarray:
        cm = self.confusion_matrix
        recall = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if cm[1, 1] > 0 else 0.0
        return recall

    @property
    def auroc(self) -> float:
        return self._auroc / self._count


def compute_confusion_matrix(
    gt: torch.Tensor, pred: torch.Tensor, num_classes: int
) -> torch.Tensor:
    gt_onehot = torch.nn.functional.one_hot(gt, num_classes).to(torch.float32)
    pred_onehot = torch.nn.functional.one_hot(pred, num_classes).to(torch.float32)
    confusion_matrix = torch.matmul(pred_onehot.t(), gt_onehot)
    return confusion_matrix
