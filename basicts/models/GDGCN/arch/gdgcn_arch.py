import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..config.gdgcn_config import GDGCNConfig


def _get_time_index(inputs_timestamps: Optional[torch.Tensor], steps_per_day: int) -> int:
    """Convert normalized timestamps to discrete time slot index.

    Args:
        inputs_timestamps (torch.Tensor | None): Timestamp tensor with shape [B, T, time_dim].
        steps_per_day (int): Number of discrete slots per day.

    Returns:
        int: Discrete time index used by dynamic graph constructors.
    """
    if inputs_timestamps is None:
        return 0
    time_feature = inputs_timestamps[0, -1, 0]
    index = int((time_feature * steps_per_day).round().item())
    return max(0, min(index, steps_per_day - 1))


class NConv(nn.Module):
    """Basic neighborhood convolution using a provided adjacency matrix."""

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N, T], adj: [N, N]
        return torch.einsum("bcnt,nm->bcmt", x, adj).contiguous()


class PointwiseConv(nn.Module):
    """1x1 convolution used for feature transformations."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Conv2DBlock(nn.Module):
    """Conv2d block with optional activation and batch normalization."""

    def __init__(self, input_dims: int, output_dims: int, kernel_size: int,
                 stride: tuple[int, int] = (1, 1), padding: str = "SAME",
                 activation=F.relu, bn_decay: float | None = None,
                 use_bias: bool = True) -> None:
        super().__init__()
        self.activation = activation
        if padding == "SAME":
            padding_size = math.ceil(kernel_size)
            self.padding = (padding_size, padding_size, padding_size, padding_size)
        else:
            self.padding = (0, 0, 0, 0)
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        nn.init.xavier_uniform_(self.conv.weight)
        if use_bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, self.padding)
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FullyConnected(nn.Module):
    """Stacked pointwise convolutions used inside attention."""

    def __init__(self, input_dims, units, activations, bn_decay: float | None, use_bias: bool = True):
        super().__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        self.convs = nn.ModuleList([
            Conv2DBlock(input_dim, num_unit, kernel_size=1, stride=(1, 1),
                        padding="VALID", use_bias=use_bias, activation=activation,
                        bn_decay=bn_decay)
            for input_dim, num_unit, activation in zip(input_dims, units, activations)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x)
        return x


class TemporalAttention(nn.Module):
    """Multi-head temporal attention treating time steps as tokens."""

    def __init__(self, layers: int, device: torch.device, K: int = 2, d: int = 4,
                 bn_decay: float = 0.1, mask: bool = False) -> None:
        super().__init__()
        D = K * d
        self.d = d
        self.K = K
        self.mask = mask
        self.device = device

        self.fc_q = FullyConnected(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.fc_k = FullyConnected(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.fc_v = FullyConnected(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.fc = FullyConnected(input_dims=D, units=D * 2, activations=F.relu, bn_decay=bn_decay)

    def forward(self, x: torch.Tensor, time_ind: int) -> torch.Tensor:  # noqa: ARG002
        batch_size = x.shape[0]
        x = x.transpose(1, 3).contiguous()
        query = self.fc_q(x)
        key = self.fc_k(x)
        value = self.fc_v(x)

        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        attention = torch.matmul(query, key) / (self.d ** 0.5)
        if self.mask:
            num_step = x.shape[1]
            num_vertex = x.shape[2]
            mask = torch.tril(torch.ones(num_step, num_step, device=self.device)).bool()
            mask = mask.unsqueeze(0).unsqueeze(0).repeat(self.K * batch_size, num_vertex, 1, 1)
            attention = torch.where(mask, attention, -2 ** 15 + 1)
        attention = F.softmax(attention, dim=-1)

        x = torch.matmul(attention, value)
        x = x.permute(0, 2, 1, 3)
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)
        x = self.fc(x)
        x = x.transpose(1, 3).contiguous()
        return x


class MLPTemporal(nn.Module):
    """Temporal modeling with 1x1 convolution."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor, time_ind: int, layer: int) -> torch.Tensor:  # noqa: ARG002
        x = x.transpose(1, 3).contiguous()
        x = self.mlp(x)
        return x.transpose(1, 3).contiguous()


class MLPTemporalNew(nn.Module):
    """Temporal modeling with learnable linear transformation."""

    def __init__(self, c_in: int, c_out: int, sharing_vector_dim: int, device: torch.device) -> None:  # noqa: ARG002
        super().__init__()
        self.W = nn.Parameter(torch.randn(c_in, c_out, device=device), requires_grad=True)
        self.B = nn.Parameter(torch.randn(c_out, device=device), requires_grad=True)

    def forward(self, x: torch.Tensor, time_ind: int) -> torch.Tensor:  # noqa: ARG002
        x = x.transpose(1, 3).contiguous()
        x = torch.einsum("df, bdnt->bfnt", self.W, x) + self.B.view(1, -1, 1, 1)
        return x.transpose(1, 3).contiguous()


class LSTMTemporal(nn.Module):
    """Temporal modeling using a single-layer RNN as in the ablation study."""

    def __init__(self, num_nodes: int, device: torch.device, hidden: int, layers: int) -> None:  # noqa: ARG002
        super().__init__()
        self.device = device
        self.rnn = nn.RNN(hidden, hidden, 1).to(device)

    def forward(self, x: torch.Tensor, time_ind: int) -> torch.Tensor:  # noqa: ARG002
        B, D, N, T = x.shape
        x = x.permute(3, 0, 2, 1).reshape(T, B * N, D)
        h0 = torch.zeros((1, B * N, D), device=self.device)
        output, _ = self.rnn(x, h0)
        x = output.reshape(T, B, N, D).permute(1, 3, 2, 0)
        return x


class TemporalConvNet(nn.Module):
    """Temporal convolution used in the ablation study."""

    def __init__(self, residual_channels: int, dilation_channels: int, kernel_size: int, layers: int, new_dilation: int) -> None:  # noqa: ARG002
        super().__init__()
        self.filter_convs = nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation)
        self.gate_convs = nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation)

    def forward(self, x: torch.Tensor, time_ind: int) -> torch.Tensor:  # noqa: ARG002
        original_size = x.size(3)
        filt = torch.tanh(self.filter_convs(x))
        gate = torch.sigmoid(self.gate_convs(x))
        x = filt * gate
        x = F.pad(x, (original_size - x.size(3), 0, 0, 0))
        return x


class TemporalGraphConv(nn.Module):
    """Graph convolution over temporal dimension using dynamic adjacency."""

    def __init__(self, dropout: float, num_steps: int, inter_dim: int) -> None:
        super().__init__()
        self.nconv = NConv()
        self.dropout = dropout
        self.timevec1 = nn.Parameter(torch.randn(num_steps, inter_dim))
        self.timevec2 = nn.Parameter(torch.randn(num_steps, inter_dim))
        self.nodevec = nn.Parameter(torch.randn(288, inter_dim))
        self.k = nn.Parameter(torch.randn(inter_dim, inter_dim, inter_dim))

    def forward(self, x: torch.Tensor, time_ind: int) -> torch.Tensor:
        x = x.transpose(2, 3).contiguous()  # [B, C, T, N]
        # time-dependent vector should keep two dims for einsum
        time_vec = self.nodevec[time_ind].unsqueeze(0)
        adp1 = torch.einsum("ad,def->aef", time_vec, self.k)
        adp2 = torch.einsum("be,aef->abf", self.timevec1, adp1)
        adp3 = torch.einsum("cf,abf->abc", self.timevec2, adp2)
        adp = F.softmax(F.relu(adp3), dim=2)
        x = self.nconv(x, adp)
        x = F.dropout(x, self.dropout, training=self.training)
        return x.transpose(2, 3).contiguous()


class SpatialGraphConv(nn.Module):
    """Dynamic spatial graph convolution using Tucker-style decomposition."""

    def __init__(self, dropout: float, num_nodes: int, inter_dim: int) -> None:
        super().__init__()
        self.nconv = NConv()
        self.dropout = dropout
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, inter_dim))
        self.nodevec2 = nn.Parameter(torch.randn(num_nodes, inter_dim))
        self.timevec = nn.Parameter(torch.randn(288, inter_dim))
        self.k = nn.Parameter(torch.randn(inter_dim, inter_dim, inter_dim))

    def forward(self, x: torch.Tensor, time_ind: int) -> torch.Tensor:
        # preserve 2D shape for the time-dependent factor in einsum
        time_vec = self.timevec[time_ind].unsqueeze(0)
        adp1 = torch.einsum("ad,def->aef", time_vec, self.k)
        adp2 = torch.einsum("be,aef->abf", self.nodevec1, adp1)
        adp3 = torch.einsum("cf,abf->abc", self.nodevec2, adp2)
        adp = F.softmax(F.relu(adp3), dim=2)
        x = self.nconv(x, adp)
        return F.dropout(x, self.dropout, training=self.training)


class GDGCN(nn.Module):
    """Graph Dynamics for Spatial-Temporal Traffic Forecasting (GDGCN)."""

    def __init__(self, config: GDGCNConfig) -> None:
        super().__init__()
        self.dropout = config.dropout
        self.layers = config.num_layers
        self.temporal_mode = config.temporal_mode
        sharing_vector_dim = self.layers

        self.spatial = SpatialGraphConv(config.dropout, config.num_features, config.spatial_inter_dim)

        if self.temporal_mode == "tcn":
            self.temporal = TemporalConvNet(config.residual_channels, config.dilation_channels,
                                            config.kernel_size, self.layers, new_dilation=1)
        elif self.temporal_mode == "lstm":
            self.temporal = LSTMTemporal(config.num_features, torch.device("cpu"),
                                         config.dilation_channels, self.layers)
        elif self.temporal_mode == "attention":
            self.temporal = TemporalAttention(self.layers, device=torch.device("cpu"))
        elif self.temporal_mode == "mlp":
            self.temporal = MLPTemporal(config.input_len, config.input_len)
        elif self.temporal_mode == "mlp_new":
            self.temporal = MLPTemporalNew(config.input_len, config.input_len, sharing_vector_dim, torch.device("cpu"))
        else:
            self.temporal = TemporalGraphConv(config.dropout, config.input_len, config.temporal_inter_dim)

        self.feature = PointwiseConv(config.dilation_channels, config.residual_channels)

        self.layer_spatial = nn.ModuleList()
        self.layer_temporal = nn.ModuleList()
        self.layer_feature = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.result_fuse = nn.ModuleList()

        for _ in range(self.layers):
            if self.temporal_mode == "tcn":
                temporal_module = TemporalConvNet(config.residual_channels, config.dilation_channels,
                                                  config.kernel_size, self.layers, new_dilation=1)
            elif self.temporal_mode == "lstm":
                temporal_module = LSTMTemporal(config.num_features, torch.device("cpu"),
                                               config.dilation_channels, self.layers)
            elif self.temporal_mode == "attention":
                temporal_module = TemporalAttention(self.layers, device=torch.device("cpu"))
            elif self.temporal_mode == "mlp":
                temporal_module = MLPTemporal(config.input_len, config.input_len)
            elif self.temporal_mode == "mlp_new":
                temporal_module = MLPTemporalNew(config.input_len, config.input_len, sharing_vector_dim, torch.device("cpu"))
            else:
                temporal_module = TemporalGraphConv(config.dropout, config.input_len, config.temporal_inter_dim)

            self.layer_spatial.append(SpatialGraphConv(config.dropout, config.num_features, config.spatial_inter_dim))
            self.layer_temporal.append(temporal_module)
            self.layer_feature.append(PointwiseConv(config.dilation_channels, config.residual_channels))

            self.residual_convs.append(nn.Conv2d(config.dilation_channels, config.residual_channels, kernel_size=(1, 1)))
            self.skip_convs.append(nn.Conv2d(config.dilation_channels, config.skip_channels, kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(config.residual_channels))
            if self.temporal_mode == "no_temporal":
                fuse_channels = config.dilation_channels * 4
            else:
                fuse_channels = config.dilation_channels * 6
            self.result_fuse.append(nn.Conv2d(fuse_channels, config.residual_channels, kernel_size=(1, 1)))

        self.start_conv = nn.Conv2d(in_channels=config.input_dim, out_channels=config.residual_channels, kernel_size=(1, 1))
        self.end_conv_1 = nn.Conv2d(in_channels=config.skip_channels, out_channels=config.residual_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=config.output_len * config.residual_channels, out_channels=config.output_len, kernel_size=(1, 1), bias=True)

    def forward(self, inputs: torch.Tensor, inputs_timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs (torch.Tensor): Input tensor with shape [B, P, N, C].
            inputs_timestamps (torch.Tensor | None): Timestamps with shape [B, P, time_dim].

        Returns:
            torch.Tensor: Prediction with shape [B, Q, N].
        """
        time_ind = _get_time_index(inputs_timestamps, steps_per_day=self.spatial.timevec.shape[0])
        # Ensure dense 4D input: [B, P, N, C]
        if inputs.is_sparse:
            inputs = inputs.to_dense()
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(-1)
        if inputs.dim() != 4:
            raise ValueError(f"GDGCN expects 4D input [B, P, N, C], but got shape {tuple(inputs.shape)}")

        x = inputs.permute(0, 3, 2, 1).contiguous()
        x = self.start_conv(x)
        skip = 0
        for i in range(self.layers):
            residual = x
            spatial_a = self.spatial(x, time_ind)
            temporal_a = self.temporal(x, time_ind)
            feature_a = self.feature(x)

            spatial_b = self.layer_spatial[i](x, time_ind)
            temporal_b = self.layer_temporal[i](x, time_ind)
            feature_b = self.layer_feature[i](x)

            if self.temporal_mode == "no_temporal":
                x = torch.cat([spatial_a, feature_a, spatial_b, feature_b], dim=1)
            else:
                x = torch.cat([spatial_a, spatial_b, temporal_a, temporal_b, feature_a, feature_b], dim=1)

            x = F.relu(x)
            x = self.result_fuse[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
            s = self.skip_convs[i](x)
            skip = s + skip

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = x.transpose(2, 3)
        x = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3), 1)
        x = self.end_conv_2(x)
        # [B, Q, N, 1]
        return x.squeeze(-1).transpose(1, 2)
