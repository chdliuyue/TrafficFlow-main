import torch
from torch import nn
from einops.layers.torch import Rearrange


class SpatioTemporalAttention(nn.Module):
    """Spatial-temporal attention that mixes spatial channel statistics."""

    def __init__(self) -> None:
        super().__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode="reflect", bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        return self.sa(x2)


class ChannelAttention(nn.Module):
    """Channel attention using squeeze-and-excitation style pooling."""

    def __init__(self, dim: int, reduction: int = 8) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gap = self.gap(x)
        return self.ca(x_gap)


class PixelAttention(nn.Module):
    """Pixel-wise attention guided by spatial-temporal cues."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode="reflect", groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, pattn1: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(dim=2)
        pattn1 = pattn1.unsqueeze(dim=2)
        x2 = torch.cat([x, pattn1], dim=2)
        x2 = Rearrange("b c h t n -> b (c h) t n")(x2)
        pattn2 = self.pa2(x2)
        return self.sigmoid(pattn2)


class CGAFusion(nn.Module):
    """Channel-Gated Adaptive Fusion for spatial and temporal branches."""

    def __init__(self, dim: int, reduction: int = 8) -> None:
        super().__init__()
        self.sa = SpatioTemporalAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 3)
        y = y.transpose(1, 3)
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result.transpose(1, 3)
