from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet_blocks import DoubleConv, Down, Out, Up


class UNetMultiTask(nn.Module):
    """
    Multi-task U-Net,
    4-stage encoder and decoder path for segmentation,
    fully connected head for classification
    """

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 3,
        start_filter: int = 4,
    ) -> None:
        super(UNetMultiTask, self).__init__()
        self.n_channels = num_channels
        self.n_classes = num_classes

        # Encoder path
        self.inc = DoubleConv(num_channels, start_filter)
        self.down1 = Down(start_filter, start_filter * 2)
        self.down2 = Down(start_filter * 2, start_filter * 2**2)
        self.down3 = Down(start_filter * 2**2, start_filter * 2**3)
        self.down4 = Down(start_filter * 2**3, start_filter * 2**4)

        # Segmentation decoder path
        self.up1 = Up(start_filter * 2**4, start_filter * 2**3)
        self.up2 = Up(start_filter * 2**3, start_filter * 2**2)
        self.up3 = Up(start_filter * 2**2, start_filter * 2)
        self.up4 = Up(start_filter * 2, start_filter)
        self.out_sgm = Out(start_filter, num_classes)

        # Classification decoder path
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(start_filter * 2**4, start_filter * 2**2, bias=False)
        self.bnfc = nn.BatchNorm1d(start_filter * 2**2)
        self.out_tum_det = nn.Linear(start_filter * 2**2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Segmentation decoder path
        x_sgm = self.up1(x5, x4)
        x_sgm = self.up2(x_sgm, x3)
        x_sgm = self.up3(x_sgm, x2)
        x_sgm = self.up4(x_sgm, x1)
        sgm_logits = self.out_sgm(x_sgm)

        # Classification decoder path
        x_tum_det = torch.flatten(self.pool(x5), start_dim=1)
        x_tum_det = F.relu(self.bnfc(self.fc(x_tum_det)))
        tum_det_logits = self.out_tum_det(x_tum_det)
        return sgm_logits, tum_det_logits
