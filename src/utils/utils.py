import datetime
import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from utils.config_templates.train import TrainConfig
from utils.constants import SGM_CLASS_ID_TO_COLOR, TUM_DET_CLASS_ID_TO_COLOR


def convert_segmentation_mask_to_rgb(
    num_mask: np.ndarray, bgr: bool = False, tum_det: bool = False
) -> np.ndarray:
    rgb_mask = np.zeros((num_mask.shape[0], num_mask.shape[1], 3))
    class_id_color_map = TUM_DET_CLASS_ID_TO_COLOR if tum_det else SGM_CLASS_ID_TO_COLOR

    for class_id, color in class_id_color_map.items():
        rgb_mask[num_mask == class_id] = color
        # print(class_id, color, rgb_mask[num_mask == class_id].size)
    # print(rgb_mask.shape)
    rgb_mask = (rgb_mask * 255).astype("uint8")
    if bgr:
        rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)
    return rgb_mask


def create_input_output_mapping(depth: int = 5, max_enc_dim: int = 200) -> int:
    input_dims = []
    output_dims = []
    for encoded_dim in range(5, max_enc_dim):
        input_dim = encoded_dim + 4
        output_dim = encoded_dim
        for _ in range(depth - 1):
            input_dim = (input_dim * 2) + 4
            output_dim = (output_dim * 2) - 4
        input_dims.append(input_dim)
        output_dims.append(output_dim)
    return input_dims, output_dims


def get_input_from_output_size(
    output_size: int, depth: int = 5, verbose: bool = True
) -> Tuple[int, int]:
    input_dims, output_dims = create_input_output_mapping(depth)
    input_dims, output_dims = np.array(input_dims), np.array(output_dims)
    nearest_value = output_dims.flat[np.abs(output_dims - output_size).argmin()]
    if verbose and nearest_value != output_size:
        print(
            f"Desired output size {output_size} is not possible, nearest possible output size is {nearest_value}!"
        )
    return int(input_dims[output_dims == nearest_value]), nearest_value


def create_output_folder(config: TrainConfig) -> str:
    current_datetime = datetime.datetime.now()
    print(current_datetime)
    current_date = (
        f"{current_datetime.day}_{current_datetime.month}_{current_datetime.year}"
    )
    current_time = (
        f"{current_datetime.hour}_{current_datetime.minute}_{current_datetime.second}"
    )

    output_path = str(Path(config.paths.output_path, f"{current_date}_{current_time}"))
    os.makedirs(output_path)
    return output_path


def pretty_print_confusion_matrix(
    conf_matrix: np.ndarray, class_names: List[str]
) -> str:
    # Set the width of each column to the length of the longest class name
    col_width = len(str(np.max(conf_matrix))) + 3

    # Build the output string
    output = ""
    output += " " * col_width
    output += "Ground Truth".center(col_width * len(class_names)) + "\n"
    output += " " * col_width
    for name in class_names:
        output += "%{0}s".format(col_width) % name
    output += "\n"
    for i, name in enumerate(class_names):
        output += "%{0}s".format(col_width) % name
        for j in range(conf_matrix.shape[1]):
            count = conf_matrix[i, j]
            output += "%{0}d".format(col_width) % count
        output += "\n"
    output += "\n"
    return output


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
