import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sn
import pandas as pd
from utils.constants import SGM_CLASSES, SGM_CLASS_ID_TO_COLOR
from utils.utils import convert_segmentation_mask_to_rgb

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150


def center_crop(img: np.ndarray, crop_size: int) -> np.ndarray:
    center = img.shape
    x = center[1] / 2 - crop_size / 2
    y = center[0] / 2 - crop_size / 2

    crop_img = img[int(y) : int(y + crop_size), int(x) : int(x + crop_size)]
    return crop_img


def plot_batch_gallery(
    images: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    num_cols: int = 8,
) -> None:
    batch_size = images.shape[0]
    num_rows = (batch_size + num_cols - 1) // num_cols
    masks = masks.detach().int().cpu().squeeze(dim=1).numpy()
    images = images.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(int(num_cols * 1.3), int((num_rows + int(num_rows * 0.6)) * 1.5)),
    )
    for i, ax in enumerate(axs.flat):
        if i < batch_size:
            image, mask, label = images[i], masks[i], labels[i]
            image = (image * 255).astype("uint8")
            image = np.moveaxis(image, 0, -1)
            mask = mask + 1

            # mask = combine_masks(masks[i], C=NUM_SGM_CLASSES - 1)
            rgb_mask = convert_segmentation_mask_to_rgb(mask)
            if image.shape != rgb_mask.shape:
                print(f"Shape mismatch! image: {image.shape}, mask: {rgb_mask.shape}")
            overlay = cv2.addWeighted(image, 0.7, rgb_mask, 0.3, 0)
            ax.imshow(overlay)
            if type(label) == np.float32:
                ax.set_title(f"Label: {label:.2f}")
            else:
                ax.set_title(f"Label: {label}")
        ax.axis("off")

    fig.legend(
        [
            plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor="black")
            for color in SGM_CLASS_ID_TO_COLOR.values()
        ],
        SGM_CLASSES,
        bbox_to_anchor=(0.5, -0.01),
        loc="lower center",
        ncol=6,
        prop={"size": num_cols},
    )
    # plt.tight_layout()
    return fig


def plot_sgm_cm(sgm_cm: np.ndarray, norm: str) -> plt.Axes:
    if norm == "all":
        sgm_cm = sgm_cm.astype("float") / np.sum(sgm_cm)
    elif norm == "true":
        sgm_cm = sgm_cm.astype("float") / sgm_cm.sum(axis=0, keepdims=1)

    sgm_df = pd.DataFrame(sgm_cm, columns=SGM_CLASSES[1:], index=SGM_CLASSES[1:])
    sgm_cm_fig, ax = plt.subplots(figsize=(7, 7))
    sn.heatmap(sgm_df, cmap="Blues", annot=True, square=True, cbar=False, fmt=".2f")
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )
    ax.set_xlabel("Ground Truth", loc="center")
    ax.set_ylabel("Prediction", loc="center")
    sgm_cm_fig.set_tight_layout(True)
    return sgm_cm_fig


def plot_tum_det_cm(tum_det_cm: np.ndarray, norm: str = "") -> plt.Axes:
    if norm == "all":
        tum_det_cm = tum_det_cm.astype("float") / np.sum(tum_det_cm)
    elif norm == "true":
        tum_det_cm = tum_det_cm.astype("float") / tum_det_cm.sum(axis=0)[:, np.newaxis]

    tum_det_df = pd.DataFrame(
        tum_det_cm, columns=["No tumor", "Tumor"], index=["No tumor", "Tumor"]
    )
    tum_det_cm_fig, ax = plt.subplots(figsize=(4, 4))
    sn.heatmap(tum_det_df, cmap="Blues", annot=True, square=True, cbar=False, fmt=".2f")
    ax.set_xlabel("Ground Truth", loc="center")
    ax.set_ylabel("Prediction", loc="center")
    tum_det_cm_fig.set_tight_layout(True)
    return tum_det_cm_fig


def plot_probability_hist(
    pos_hist: np.ndarray, neg_hist: np.ndarray, num_bins: int = 100
) -> plt.Axes:
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    fig, ax = plt.subplots()
    ax.bar(
        bin_edges[:-1],
        neg_hist,
        width=1 / num_bins,
        alpha=0.5,
        color="red",
        label="no_tumor",
    )
    ax.bar(
        bin_edges[:-1],
        pos_hist,
        width=1 / num_bins,
        alpha=0.5,
        color="blue",
        label="tumor",
    )
    ax.set_xlabel("Probability value bin")
    ax.set_ylabel("Num of samples in bin")
    ax.legend()
    fig.set_tight_layout(True)
    return fig


def visualize_f1_scores(f1_scores: np.ndarray) -> plt.Axes:
    """Visualizes f1 scores per class."""

    fig, ax = plt.subplots(figsize=(10, 6))
    df = pd.DataFrame([f1_scores], columns=SGM_CLASSES[1:])

    # create the heatmap using Seaborn
    sn.heatmap(
        df,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        annot=True,
        square=True,
        fmt=".3f",
        linewidths=0.5,
        ax=ax,
        cbar=False,
    )
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )
    ax.set(ylabel=None)
    # customize the plot
    ax.set_title("F1 Scores per Class")
    fig.set_tight_layout(True)
    return fig
