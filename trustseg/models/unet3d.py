from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(
            inputs,
            size=skip.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        return self.conv(torch.cat((skip, inputs), dim=1))


class UNet3D(nn.Module):
    """3D U-Net used for both the initial and final TRUST-Seg students."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 16,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        channels = [base_channels * (2**index) for index in range(5)]

        self.enc1 = ConvBlock(in_channels, channels[0])
        self.enc2 = ConvBlock(channels[0], channels[1])
        self.enc3 = ConvBlock(channels[1], channels[2])
        self.enc4 = ConvBlock(channels[2], channels[3])
        self.center = ConvBlock(channels[3], channels[4])
        self.pool = nn.MaxPool3d(kernel_size=2)

        self.up4 = UpBlock(channels[4], channels[3], channels[3])
        self.up3 = UpBlock(channels[3], channels[2], channels[2])
        self.up2 = UpBlock(channels[2], channels[1], channels[1])
        self.up1 = UpBlock(channels[1], channels[0], channels[0])
        self.dropout = nn.Dropout3d(p=dropout)
        self.output = nn.Conv3d(channels[0], out_channels, kernel_size=1)

        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(inputs)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        center = self.dropout(self.center(self.pool(enc4)))

        decoded = self.up4(center, enc4)
        decoded = self.up3(decoded, enc3)
        decoded = self.up2(decoded, enc2)
        decoded = self.dropout(self.up1(decoded, enc1))
        return self.output(decoded)

