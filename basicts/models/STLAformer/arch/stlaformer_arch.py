import torch
from torch import nn

from ..config.stlaformer_config import STLAformerConfig
from .positional_encoding import NodePositionalEncoding, TimePositionalEncoding
from .stla_layer import STLAEncoderLayer


class STLAformer(nn.Module):
    """Implementation of the STLAformer for spatial-temporal forecasting."""

    def __init__(self, config: STLAformerConfig):
        super().__init__()
        self.num_nodes = config.num_features
        self.in_steps = config.input_len
        self.out_steps = config.output_len
        self.steps_per_week = config.steps_per_week
        self.input_dim = 1
        self.input_embedding_dim = config.input_embedding_dim
        self.tow_embedding_dim = config.tow_embedding_dim
        self.model_dim = self.input_embedding_dim + self.tow_embedding_dim
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers

        if self.num_nodes is None or self.in_steps is None or self.out_steps is None:
            raise ValueError("STLAformer requires `num_features`, `input_len`, and `output_len` to be specified in the config.")

        self.input_proj = nn.Linear(self.input_dim, self.input_embedding_dim)
        if self.tow_embedding_dim > 0:
            self.tow_embedding = nn.Embedding(self.steps_per_week, self.tow_embedding_dim)
        else:
            self.tow_embedding = None

        self.output_proj = nn.Linear(self.in_steps * self.model_dim, self.out_steps)
        self.attn_layers_st = nn.ModuleList(
            [
                STLAEncoderLayer(self.model_dim, config.feed_forward_dim, self.num_heads, config.dropout)
                for _ in range(self.num_layers)
            ]
        )
        self.node_position_encoding = NodePositionalEncoding(self.model_dim)
        self.time_position_encoding = TimePositionalEncoding(self.model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.model_dim, self.model_dim),
        )

    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs (torch.Tensor): Input tensor with shape [batch_size, in_steps, num_nodes].
            inputs_timestamps (torch.Tensor | None): Optional timestamps with shape [batch_size, in_steps, time_feat_dim].

        Returns:
            torch.Tensor: Prediction with shape [batch_size, out_steps, num_nodes].
        """

        batch_size = inputs.shape[0]
        x = self.input_proj(inputs.unsqueeze(-1))
        features = [x]

        if self.tow_embedding is not None:
            if inputs_timestamps is not None:
                tow_feature = inputs_timestamps[..., 0]
                tow_indices = (tow_feature * self.steps_per_week).long() % self.steps_per_week
                tow_emb = self.tow_embedding(tow_indices)
                tow_emb = tow_emb.unsqueeze(-2).expand(-1, -1, self.num_nodes, -1)
            else:
                tow_emb = torch.zeros(
                    batch_size,
                    self.in_steps,
                    self.num_nodes,
                    self.tow_embedding_dim,
                    device=inputs.device,
                    dtype=x.dtype,
                )
            features.append(tow_emb)
        x = torch.cat(features, dim=-1)

        node_emb = self.node_position_encoding(x)
        time_emb = self.time_position_encoding(x)

        baseline = self.mlp(x)
        x = x - baseline
        for attn in self.attn_layers_st:
            x = attn(x, node_emb, time_emb)
        x = x + baseline

        out = x.transpose(1, 2)
        out = out.reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
        out = self.output_proj(out).view(batch_size, self.num_nodes, self.out_steps)
        out = out.transpose(1, 2)
        return out
