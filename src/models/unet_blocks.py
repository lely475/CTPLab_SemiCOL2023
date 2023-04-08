from typing import Tuple

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """2x (3x3 conv -> ReLU -> BN)"""

    def __init__(self, in_size: int, out_size: int):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
            # bias False due to following Batch Norm layer
            nn.Conv2d(out_size, out_size, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block:
    (1) max pool 2x2, stride 2
    (2) 2x 3x3 conv+ReLU+BatchNorm"""

    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2, stride=2), DoubleConv(in_size, out_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block"""

    def __init__(self, in_size: int, out_size: int) -> None:
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        self.conv_block = DoubleConv(in_size, out_size)

    def center_crop(self, x: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
        _, _, h, w = x.size()
        diff_y = torch.div(h - out_size[0], 2, rounding_mode="floor")
        diff_x = torch.div(w - out_size[1], 2, rounding_mode="floor")
        return x[:, :, diff_y : (diff_y + out_size[0]), diff_x : (diff_x + out_size[1])]

    def forward(self, x: torch.Tensor, down_equiv: torch.Tensor) -> torch.Tensor:
        up = self.up(x)
        crop = self.center_crop(down_equiv, up.shape[2:])
        out = torch.cat([up, crop], 1)
        out = self.conv_block(out)
        return out


class Out(nn.Module):
    """1x1 convolution to get desired output channels"""

    def __init__(self, in_size: int, out_size: int) -> None:
        super(Out, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
