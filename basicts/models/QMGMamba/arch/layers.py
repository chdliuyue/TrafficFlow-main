"""Core layers for QMGMamba.

This file contains:
- TimestampEmbedding: embedding of categorical timestamps (e.g., time-of-day, day-of-week)
- QMG adjacency generator: learns a (possibly dynamic) *complex* directed graph
- QMGConvBlock: complex graph propagation with optional signed edges
- Temporal backbones: Mamba2-like selective SSM, TransformerEncoder, GRU/LSTM
- ForecastHead: horizon-wise decoding (optionally conditioned on future timestamps)

Design goals:
- Compatible with BasicTS forward signature:
    forward(inputs, targets, inputs_timestamps, targets_timestamps)
- All important switches are in config.

Note on "complex graph":
We represent a complex edge weight as w_ij * exp(i * phi_ij).
In implementation we keep (A_re, A_im) where:
    A_re = w * cos(phi),  A_im = w * sin(phi)
and do complex message passing via real/imag decomposition.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


# =========================
# Utilities
# =========================


def _get_activation(name: str) -> nn.Module:
    name = (name or "silu").lower()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


def _topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Return a {0,1} mask keeping top-k (by absolute value) per row.

    Args:
        scores: [B, N, N]
        k: top-k

    Returns:
        mask: [B, N, N] float tensor in {0,1}
    """
    if k <= 0 or k >= scores.size(-1):
        return torch.ones_like(scores)
    # absolute scores for selection
    _, idx = torch.topk(scores.abs(), k=k, dim=-1)
    mask = torch.zeros_like(scores)
    mask.scatter_(-1, idx, 1.0)
    return mask


# =========================
# Timestamp embedding
# =========================


class TimestampEmbedding(nn.Module):
    """Embed categorical timestamps.

    Typical PeMS timestamps in BasicTS are 2-dimensional:
    - time of day index: 0..287
    - day of week index: 0..6

    Inputs:
        ts: [B, L, T] (integer)

    Outputs:
        emb: [B, L, D]
    """

    def __init__(self, timestamp_sizes, d_model: int, dropout: float = 0.0):
        super().__init__()
        if timestamp_sizes is None:
            raise ValueError("timestamp_sizes must be provided when use_timestamp=True")
        self.embeddings = nn.ModuleList([
            nn.Embedding(int(size), d_model) for size in timestamp_sizes
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, ts: torch.Tensor) -> torch.Tensor:
        # ts: [B, L, T]
        if ts is None:
            raise ValueError("timestamps tensor is None but TimestampEmbedding is enabled")
        if ts.dtype not in (torch.int32, torch.int64):
            ts = ts.long()
        out = 0.0
        for i, emb in enumerate(self.embeddings):
            out = out + emb(ts[..., i])
        return self.dropout(out)


# =========================
# QMG adjacency & propagation
# =========================


class QMGAdjacency(nn.Module):
    """Learn a directed (possibly dynamic) complex adjacency.

    We parameterize base (static) edge logits and phase logits with node embeddings:
        amp_logits(i,j) = <e_src(i), e_dst(j)>
        phi_base(i,j)   = pi * tanh(<p_src(i), p_dst(j)>)  in [-pi, pi]

    If dynamic graph is enabled, we modulate these logits by a state-dependent term
    derived from node context features (e.g., last hidden state):
        amp_logits_b(i,j) = amp_logits(i,j) + s * g_src_b(i) * g_dst_b(j)
        phi_b(i,j)        = phi_base(i,j) + s_phi * (q_src_b(i) - q_dst_b(j))

    Finally we map amp_logits to signed/nonnegative weights and normalize row-wise.

    Args:
        num_nodes: N
        d_emb: embedding dimension for graph parameterization
        dynamic: state-dependent graph
        forbid_negative_edges: if True, enforce nonnegative row-stochastic weights
        use_complex: if True, output (A_re, A_im) using phase; else, A_im=0
    """

    def __init__(
        self,
        num_nodes: int,
        d_emb: int,
        ctx_dim: Optional[int] = None,
        dynamic: bool = True,
        forbid_negative_edges: bool = False,
        use_complex: bool = True,
        temperature: float = 1.0,
        topk: int = 20,
        add_self_loops: bool = True,
        self_loop_weight: float = 0.2,
        dropout: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.dynamic = bool(dynamic)
        self.forbid_negative_edges = bool(forbid_negative_edges)
        self.use_complex = bool(use_complex)
        self.temperature = float(temperature)
        self.topk = int(topk)
        self.add_self_loops = bool(add_self_loops)
        self.self_loop_weight = float(self_loop_weight)
        self.eps = float(eps)

        # base embeddings to produce directed edge logits (asymmetric)
        self.e_src = nn.Parameter(torch.randn(self.num_nodes, self.d_emb) * 0.02)
        self.e_dst = nn.Parameter(torch.randn(self.num_nodes, self.d_emb) * 0.02)

        # base embeddings to produce phase logits
        self.p_src = nn.Parameter(torch.randn(self.num_nodes, self.d_emb) * 0.02)
        self.p_dst = nn.Parameter(torch.randn(self.num_nodes, self.d_emb) * 0.02)

        # dynamic modulators from node context
        self.dropout = nn.Dropout(dropout)

        if self.dynamic:
            if ctx_dim is None:
                raise ValueError("ctx_dim must be provided when dynamic=True")
            self._ctx_dim = int(ctx_dim)
            # amplitude gates
            self.g_src = nn.Linear(self._ctx_dim, 1)
            self.g_dst = nn.Linear(self._ctx_dim, 1)
            # phase shifts
            self.q_src = nn.Linear(self._ctx_dim, 1)
            self.q_dst = nn.Linear(self._ctx_dim, 1)
            # learnable scales
            self.amp_scale = nn.Parameter(torch.tensor(1.0))
            self.phi_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, node_ctx: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute (A_re, A_im).

        Args:
            node_ctx: [B, N, D] node context; required when dynamic=True.

        Returns:
            A_re: [B, N, N]
            A_im: [B, N, N]
            extras: dict with intermediate tensors for visualization
        """
        # base logits
        # [N, N]
        amp0 = (self.e_src @ self.e_dst.t()) / math.sqrt(self.d_emb)
        phi0 = (self.p_src @ self.p_dst.t()) / math.sqrt(self.d_emb)
        phi0 = math.pi * torch.tanh(phi0)  # [-pi, pi]

        if self.dynamic:
            if node_ctx is None:
                raise ValueError("node_ctx must be provided when omg_dynamic=True")
            B = node_ctx.size(0)
            # gates: [B, N, 1]
            g_src = torch.sigmoid(self.g_src(node_ctx))
            g_dst = torch.sigmoid(self.g_dst(node_ctx))
            # outer product -> [B, N, N]
            dyn_amp = self.amp_scale * (g_src @ g_dst.transpose(1, 2))
            amp_logits = amp0.unsqueeze(0) + dyn_amp

            q_src = torch.tanh(self.q_src(node_ctx)).squeeze(-1)  # [B, N]
            q_dst = torch.tanh(self.q_dst(node_ctx)).squeeze(-1)  # [B, N]
            # pairwise difference -> [B, N, N]
            dyn_phi = self.phi_scale * (q_src[:, :, None] - q_dst[:, None, :])
            phi = phi0.unsqueeze(0) + dyn_phi
        else:
            # static graph: broadcast to batch=1 (then expanded if needed)
            B = 1 if node_ctx is None else node_ctx.size(0)
            amp_logits = amp0.unsqueeze(0).expand(B, -1, -1)
            phi = phi0.unsqueeze(0).expand(B, -1, -1)

        # temperature
        amp_logits = amp_logits / max(self.temperature, self.eps)

        if self.forbid_negative_edges:
            # nonnegative, row-stochastic
            w = torch.softmax(amp_logits, dim=-1)
        else:
            # signed weights in [-1,1], then row-normalize by abs-sum
            w = torch.tanh(amp_logits)
            w = w / (w.abs().sum(dim=-1, keepdim=True) + self.eps)

        # sparsify
        if self.topk > 0 and self.topk < self.num_nodes:
            mask = _topk_mask(w, self.topk)
            w = w * mask
            # renormalize after masking
            if self.forbid_negative_edges:
                w = w / (w.sum(dim=-1, keepdim=True) + self.eps)
            else:
                w = w / (w.abs().sum(dim=-1, keepdim=True) + self.eps)

        # self-loops
        if self.add_self_loops:
            eye = torch.eye(self.num_nodes, device=w.device, dtype=w.dtype).unsqueeze(0)
            w = (1.0 - self.self_loop_weight) * w + self.self_loop_weight * eye
            # (optional) small renorm for stability
            if self.forbid_negative_edges:
                w = w / (w.sum(dim=-1, keepdim=True) + self.eps)
            else:
                w = w / (w.abs().sum(dim=-1, keepdim=True) + self.eps)

        w = self.dropout(w)

        if self.use_complex:
            A_re = w * torch.cos(phi)
            A_im = w * torch.sin(phi)
        else:
            A_re = w
            A_im = torch.zeros_like(w)

        extras = {
            "amp_logits": amp_logits.detach(),
            "phi": phi.detach(),
            "w": w.detach(),
        }
        return A_re, A_im, extras


class QMGConvBlock(nn.Module):
    """One QMG propagation block (pre-norm residual).

    Input:
        x: [B, L, N, D]

    Output:
        y: [B, L, N, D]

    Also returns graph tensors if requested.
    """

    def __init__(
        self,
        d_model: int,
        adjacency: QMGAdjacency,
        dropout: float = 0.0,
        activation: str = "silu",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.adj = adjacency
        self.norm = nn.LayerNorm(self.d_model)
        self.act = _get_activation(activation)

        # Map real features -> (re, im)
        self.to_re = nn.Linear(self.d_model, self.d_model)
        self.to_im = nn.Linear(self.d_model, self.d_model)

        # Map (re, im) back to real
        self.out = nn.Linear(2 * self.d_model, self.d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        node_ctx: Optional[torch.Tensor] = None,
        return_graph: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward.

        Args:
            x: [B, L, N, D]
            node_ctx: [B, N, D] context for dynamic graph
            return_graph: whether to return adjacency info
        """
        residual = x
        x = self.norm(x)

        A_re, A_im, extras = self.adj(node_ctx)

        x_re = self.to_re(x)
        x_im = self.to_im(x)

        # complex message passing across node dimension
        # einsum: [B,N,N] x [B,L,N,D] -> [B,L,N,D]
        msg_re = torch.einsum("bij,btjd->btid", A_re, x_re) - torch.einsum("bij,btjd->btid", A_im, x_im)
        msg_im = torch.einsum("bij,btjd->btid", A_re, x_im) + torch.einsum("bij,btjd->btid", A_im, x_re)

        msg = torch.cat([msg_re, msg_im], dim=-1)
        msg = self.out(self.act(msg))
        msg = self.dropout(msg)

        y = residual + msg

        if return_graph:
            graph = {
                "A_re": A_re.detach(),
                "A_im": A_im.detach(),
                **extras,
            }
            return y, graph
        return y, None


# =========================
# Temporal backbones
# =========================


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D_skip: torch.Tensor,
) -> torch.Tensor:
    """A simple (readable) selective scan in pure PyTorch.

    Shapes:
        u:      [B, L, D]
        delta:  [B, L, D]
        A:      [D, d_state]
        B:      [B, L, D, d_state]
        C:      [B, L, D, d_state]
        D_skip: [D]

    Returns:
        y: [B, L, D]

    This is not the most optimized implementation, but is dependency-free.
    """
    Bsz, L, Dm = u.shape
    d_state = A.size(-1)

    # state: [B, D, d_state]
    x = u.new_zeros(Bsz, Dm, d_state)
    ys = []

    # precompute A to be used with delta_t (elementwise)
    # exp(delta*A) keeps stability when A is negative
    for t in range(L):
        dt = delta[:, t, :].unsqueeze(-1)  # [B, D, 1]
        # discretized A
        A_dt = torch.exp(dt * A.unsqueeze(0))  # [B, D, d_state]
        x = A_dt * x + dt * B[:, t, :, :] * u[:, t, :].unsqueeze(-1)
        y = (C[:, t, :, :] * x).sum(dim=-1) + D_skip * u[:, t, :]
        ys.append(y)

    return torch.stack(ys, dim=1)


class Mamba2Block(nn.Module):
    """A lightweight Mamba2-like block (selective SSM + depthwise conv).

    This is a *pure PyTorch* implementation meant for research/prototyping.
    It follows the spirit of Mamba: input-dependent (selective) SSM parameters
    with a parallel local convolution path.

    Input/Output:
        x: [B, L, D]

    Reference (conceptual): Mamba / selective state space models.
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)

        self.norm = nn.LayerNorm(self.d_model)

        # gating + value
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_model)

        # depthwise conv (local mixing)
        # padding = d_conv-1 enables causal-like behavior; we will trim to length.
        self.conv = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=self.d_conv,
            groups=self.d_model,
            padding=self.d_conv - 1,
        )

        # selective SSM parameters
        self.delta_proj = nn.Linear(self.d_model, self.d_model)
        self.B_proj = nn.Linear(self.d_model, self.d_model * self.d_state)
        self.C_proj = nn.Linear(self.d_model, self.d_model * self.d_state)

        # A is learned per channel and per state; enforce negative for stability
        self.A_log = nn.Parameter(torch.randn(self.d_model, self.d_state) * 0.02)

        # skip connection term (D in SSM)
        self.D_skip = nn.Parameter(torch.ones(self.d_model))

        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)

        gate, u = self.in_proj(x_norm).chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        u = self.act(u)

        # depthwise conv path
        u_conv = self.conv(u.transpose(1, 2))
        u_conv = u_conv[:, :, : u.size(1)].transpose(1, 2)  # trim to L
        u = u_conv

        # SSM parameters from x_norm
        delta = F.softplus(self.delta_proj(x_norm))  # [B, L, D]
        B = self.B_proj(x_norm).view(x.size(0), x.size(1), self.d_model, self.d_state)
        C = self.C_proj(x_norm).view(x.size(0), x.size(1), self.d_model, self.d_state)

        A = -torch.exp(self.A_log)  # [D, d_state]

        y = selective_scan(u=u, delta=delta, A=A, B=B, C=C, D_skip=self.D_skip)
        y = self.out_proj(y)
        y = gate * y
        y = self.dropout(y)

        return residual + y


class TransformerTemporal(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.0,
        norm_first: bool = True,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        return self.encoder(x)


class RNNTTemporal(nn.Module):
    def __init__(
        self,
        backbone: str,
        d_model: int,
        num_layers: int,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.backbone = backbone.lower()
        rnn_cls = nn.GRU if self.backbone == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.out_proj = None
        if bidirectional:
            self.out_proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        y, _ = self.rnn(x)
        if self.out_proj is not None:
            y = self.out_proj(y)
        return y


class TemporalBackbone(nn.Module):
    """Apply a temporal backbone *per node*.

    Input:
        x: [B, L, N, D]

    Output:
        y: [B, L, N, D]

    Implementation detail:
        reshape to [B*N, L, D] and apply backbone.
    """

    def __init__(
        self,
        backbone: str,
        d_model: int,
        num_layers: int,
        dropout: float,
        n_heads: int = 8,
        d_ff: int = 256,
        transformer_norm_first: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        rnn_bidirectional: bool = False,
    ):
        super().__init__()
        self.backbone = backbone.lower()
        self.d_model = int(d_model)

        if self.backbone == "mamba2":
            self.net = nn.Sequential(*[
                Mamba2Block(d_model=self.d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, dropout=dropout)
                for _ in range(num_layers)
            ])
        elif self.backbone == "transformer":
            self.net = TransformerTemporal(
                d_model=self.d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                num_layers=num_layers,
                dropout=dropout,
                norm_first=transformer_norm_first,
            )
        elif self.backbone in {"gru", "lstm"}:
            self.net = RNNTTemporal(
                backbone=self.backbone,
                d_model=self.d_model,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=rnn_bidirectional,
            )
        else:
            raise ValueError(f"Unknown temporal_backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N, D = x.shape
        x2 = x.permute(0, 2, 1, 3).reshape(B * N, L, D)
        y2 = self.net(x2)
        y = y2.reshape(B, N, L, D).permute(0, 2, 1, 3)
        return y


# =========================
# Forecast head
# =========================


class ForecastHead(nn.Module):
    """Horizon-wise decoding.

    Given the last hidden state per node h_last: [B, N, D], we create horizon queries
    (from future timestamps or a learnable embedding) and predict y:
        y_{t+q} = f([h_last, time_embed(q)])

    Output:
        y: [B, H, N]
    """

    def __init__(
        self,
        d_model: int,
        horizon: int,
        dropout: float = 0.0,
        use_future_timestamp: bool = True,
        timestamp_embedder: Optional[TimestampEmbedding] = None,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.horizon = int(horizon)
        self.use_future_timestamp = bool(use_future_timestamp)
        self.ts_embedder = timestamp_embedder

        if (not self.use_future_timestamp) or (self.ts_embedder is None):
            # learnable horizon embeddings
            self.horizon_emb = nn.Parameter(torch.randn(self.horizon, self.d_model) * 0.02)
        else:
            self.horizon_emb = None

        self.mlp = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
        )

    def forward(self, h_last: torch.Tensor, targets_timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        # h_last: [B, N, D]
        B, N, D = h_last.shape

        if self.use_future_timestamp and (self.ts_embedder is not None) and (targets_timestamps is not None):
            # [B, H, D]
            t_emb = self.ts_embedder(targets_timestamps)
        else:
            # [H, D] -> [B, H, D]
            t_emb = self.horizon_emb.unsqueeze(0).expand(B, -1, -1)

        # combine: broadcast to [B, H, N, D]
        h = h_last.unsqueeze(1) + t_emb.unsqueeze(2)
        y = self.mlp(h).squeeze(-1)  # [B, H, N]
        return y
