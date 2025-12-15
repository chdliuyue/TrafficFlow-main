import torch
import torch.nn.functional as F
from torch import nn


class SelfAttention(nn.Module):
    """Standard multi-head self-attention with optional causal masking."""

    def __init__(self, model_dim: int, num_heads: int = 8, mask: bool = False) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)
        attn_score = (query @ key) / self.head_dim ** 0.5

        if self.mask:
            mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=query.device).tril()
            attn_score.masked_fill_(~mask, -torch.inf)

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = self.out_proj(out)
        return out


class InLineAttention(nn.Module):
    """Linear attention module applied independently on spatial or temporal axes."""

    def __init__(self, model_dim: int, num_heads: int = 8, mask: bool = False) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

        self.residual = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, kernel_size=1, groups=num_heads),
            nn.GELU(),
            nn.Conv1d(model_dim, model_dim * 3, kernel_size=1, groups=num_heads)
        )
        self.scale = (self.head_dim) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_nodes, channels = x.shape
        x = x.reshape(batch_size * time_steps, num_nodes, channels)

        q = self.FC_Q(x)
        k = self.FC_K(x)
        v = self.FC_V(x)

        q = q.reshape(batch_size * time_steps, num_nodes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size * time_steps, num_nodes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size * time_steps, num_nodes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        res_weight = self.residual(x.mean(dim=1).unsqueeze(dim=-1)).reshape(batch_size * time_steps * channels, 1, 3)
        kv = (k.transpose(-2, -1) * (self.scale / num_nodes) ** 0.5) @ (v * (self.scale / num_nodes) ** 0.5)
        x = q @ kv + (1 - q @ k.mean(dim=2, keepdim=True).transpose(-2, -1) * self.scale) * v.mean(dim=2, keepdim=True)

        x = x.transpose(1, 2).reshape(batch_size * time_steps, num_nodes, channels)
        v = v.transpose(1, 2).reshape(batch_size * time_steps, num_nodes, channels).permute(0, 2, 1)
        v = v.reshape(1, batch_size * time_steps * channels, num_nodes)
        residual = F.conv1d(v, res_weight, None, padding=1, groups=batch_size * time_steps * channels)
        residual = residual.reshape(batch_size * time_steps, channels, num_nodes).permute(0, 2, 1)

        out = x + residual
        out = out.reshape(batch_size, time_steps, num_nodes, channels)
        out = self.out_proj(out)
        return out
