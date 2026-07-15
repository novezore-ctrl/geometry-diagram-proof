"""Small U-Net for geometry primitive segmentation."""
from __future__ import annotations

import torch
from torch import nn

CLASSES = ("background", "solid_line", "dashed_line", "circle_or_ellipse", "point", "text")


class DSConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU6(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Block(nn.Module):
    def __init__(self, source: int, target: int):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(source, target, 3, padding=1, bias=False), nn.BatchNorm2d(target), nn.ReLU6(inplace=True))
        self.refine = DSConv(target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.project(x))


class GeometryUNet(nn.Module):
    """Compact encoder-decoder; output is [B, 6, H, W] logits."""
    def __init__(self, classes: int = len(CLASSES)):
        super().__init__()
        self.enc1 = Block(3, 24)
        self.enc2 = Block(24, 48)
        self.enc3 = Block(48, 96)
        self.bottleneck = Block(96, 160)
        self.dec3 = Block(160 + 96, 96)
        self.dec2 = Block(96 + 48, 48)
        self.dec1 = Block(48 + 24, 24)
        self.head = nn.Conv2d(24, classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        z = self.bottleneck(self.pool(c))
        d = nn.functional.interpolate(z, size=c.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec3(torch.cat((d, c), dim=1))
        d = nn.functional.interpolate(d, size=b.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec2(torch.cat((d, b), dim=1))
        d = nn.functional.interpolate(d, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.dec1(torch.cat((d, a), dim=1)))
