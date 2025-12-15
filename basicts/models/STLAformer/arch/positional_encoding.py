import math
import torch
from torch import nn


class NodePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for node dimension."""

    def __init__(self, embed_dim: int, max_len: int = 1000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(2)].unsqueeze(1).expand_as(x).detach()


class TimePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal dimension."""

    def __init__(self, embed_dim: int, max_len: int = 96) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False
        position = torch.arange(1000, 1000 + max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)].unsqueeze(1).transpose(1, 2).expand_as(x).detach()
