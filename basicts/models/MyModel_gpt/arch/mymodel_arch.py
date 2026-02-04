
import math
from typing import Dict, Optional, Tuple, List

import torch
from torch import nn

from ..config.mymodel_config import MyModelConfig


# ============================================================
# Mask helpers (same spirit as your MAE reference)
# ============================================================

def _normalize_mask(mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask = mask.float()
    mask = mask / (mask.mean() + eps)
    mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
    return mask


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    if mask is None:
        return torch.mean(x)
    m = _normalize_mask(mask, eps=eps)
    y = x * m
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.mean(y)


def masked_mae(pred: torch.Tensor, true: torch.Tensor, mask: Optional[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    return masked_mean(torch.abs(pred - true), mask, eps=eps)


def masked_mse(pred: torch.Tensor, true: torch.Tensor, mask: Optional[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    return masked_mean((pred - true) ** 2, mask, eps=eps)


def masked_huber(pred: torch.Tensor, true: torch.Tensor, mask: Optional[torch.Tensor], delta: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    err = pred - true
    abs_err = torch.abs(err)
    quad = torch.minimum(abs_err, abs_err.new_tensor(delta))
    lin = abs_err - quad
    loss = 0.5 * quad ** 2 + delta * lin
    return masked_mean(loss, mask, eps=eps)


# ============================================================
# Distribution NLLs (Part C)
# ============================================================

def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # per-element nll
    var = scale ** 2
    return 0.5 * (math.log(2.0 * math.pi) + torch.log(var) + (y - mu) ** 2 / var)


def studentt_nll(y: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
    # Student-t NLL, per-element
    # nll = log(scale) + 0.5*(df+1)*log(1 + ((y-mu)^2)/(df*scale^2)) + const(df)
    z2 = ((y - mu) / scale) ** 2
    df = torch.clamp(df, min=2.0 + 1e-6)
    const = torch.lgamma((df + 1.0) / 2.0) - torch.lgamma(df / 2.0) - 0.5 * torch.log(df * math.pi)
    return -const + torch.log(scale) + 0.5 * (df + 1.0) * torch.log1p(z2 / df)


def quantile_loss(y: torch.Tensor, q_pred: torch.Tensor, q: float) -> torch.Tensor:
    # pinball loss
    e = y - q_pred
    return torch.maximum((q - 1.0) * e, q * e)


# ============================================================
# Fourier basis utilities
# ============================================================

def _fourier_sincos(x: torch.Tensor, k: int) -> torch.Tensor:
    ang = 2.0 * math.pi * float(k) * x
    return torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)  # [...,2]


# ============================================================
# Backbone implementations
# ============================================================

class RNNBackbone(nn.Module):
    """
    Shared GRU/LSTM across nodes.
    Input:  [B,L,N] (+ optional ts_in [B,L,T]) (+ optional node_emb [N,E])
    Output: H_layers [num_layers,B,N,D]
    """
    def __init__(self, rnn_type: str, L: int, N: int, T: int, in_extra: int, D: int, layers: int, dropout: float, use_ts: bool):
        super().__init__()
        self.rnn_type = rnn_type
        self.L, self.N, self.T = int(L), int(N), int(T)
        self.D, self.layers = int(D), int(layers)
        self.use_ts = bool(use_ts)

        in_dim = 1 + (self.T if self.use_ts else 0) + int(in_extra)
        if rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=in_dim,
                hidden_size=self.D,
                num_layers=self.layers,
                dropout=float(dropout) if self.layers > 1 else 0.0,
                batch_first=True,
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=in_dim,
                hidden_size=self.D,
                num_layers=self.layers,
                dropout=float(dropout) if self.layers > 1 else 0.0,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

    def forward(self, x: torch.Tensor, ts_in: Optional[torch.Tensor], node_feat: Optional[torch.Tensor]) -> torch.Tensor:
        B, L, N = x.shape
        v = x.permute(0, 2, 1).unsqueeze(-1)  # [B,N,L,1]

        feats = [v]
        if self.use_ts:
            if ts_in is None:
                raise ValueError("use_input_timestamps=True but inputs_timestamps is None.")
            ts = ts_in.float().unsqueeze(1).expand(-1, N, -1, -1)  # [B,N,L,T]
            feats.append(ts)
        if node_feat is not None:
            # node_feat: [N,E] -> [B,N,L,E]
            nf = node_feat.unsqueeze(0).unsqueeze(2).expand(B, N, L, -1)
            feats.append(nf)

        inp = torch.cat(feats, dim=-1)          # [B,N,L,in_dim]
        inp = inp.reshape(B * N, L, inp.size(-1))

        if self.rnn_type == "gru":
            _, h = self.rnn(inp)  # [layers, B*N, D]
        else:
            _, (h, _) = self.rnn(inp)  # [layers, B*N, D]

        h = h.reshape(self.layers, B, N, self.D)  # [layers,B,N,D]
        return h


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1,max_len,d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,D]
        L = x.size(1)
        return x + self.pe[:, :L, :]


class TransformerBackbone(nn.Module):
    """
    Shared Transformer encoder across nodes.
    Input:  [B,L,N] (+ optional ts_in [B,L,T]) (+ optional node_emb [N,E])
    Output: H_layers [num_layers,B,N,D] where each layer provides the last-token rep.
    """
    def __init__(self, L: int, N: int, T: int, in_extra: int, D: int, layers: int,
                 nhead: int, ffn_ratio: float, dropout: float, norm_first: bool, use_pe: bool, use_ts: bool):
        super().__init__()
        self.L, self.N, self.T = int(L), int(N), int(T)
        self.D, self.layers = int(D), int(layers)
        self.use_ts = bool(use_ts)
        in_dim = 1 + (self.T if self.use_ts else 0) + int(in_extra)

        self.in_proj = nn.Linear(in_dim, self.D)
        self.use_pe = bool(use_pe)
        self.pe = PositionalEncoding(self.D, max_len=max(512, self.L)) if self.use_pe else None

        dim_ff = int(self.D * float(ffn_ratio))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.D,
                nhead=int(nhead),
                dim_feedforward=dim_ff,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=bool(norm_first),
            )
            for _ in range(self.layers)
        ])

    def forward(self, x: torch.Tensor, ts_in: Optional[torch.Tensor], node_feat: Optional[torch.Tensor]) -> torch.Tensor:
        B, L, N = x.shape
        v = x.permute(0, 2, 1).unsqueeze(-1)  # [B,N,L,1]

        feats = [v]
        if self.use_ts:
            if ts_in is None:
                raise ValueError("use_input_timestamps=True but inputs_timestamps is None.")
            ts = ts_in.float().unsqueeze(1).expand(-1, N, -1, -1)  # [B,N,L,T]
            feats.append(ts)
        if node_feat is not None:
            nf = node_feat.unsqueeze(0).unsqueeze(2).expand(B, N, L, -1)
            feats.append(nf)

        inp = torch.cat(feats, dim=-1)  # [B,N,L,in_dim]
        inp = inp.reshape(B * N, L, inp.size(-1))
        h = self.in_proj(inp)  # [B*N,L,D]
        if self.pe is not None:
            h = self.pe(h)

        outs: List[torch.Tensor] = []
        for blk in self.blocks:
            h = blk(h)  # [B*N,L,D]
            outs.append(h[:, -1, :])  # last token rep [B*N,D]

        H_layers = torch.stack(outs, dim=0)  # [layers,B*N,D]
        H_layers = H_layers.reshape(self.layers, B, N, self.D)
        return H_layers


def build_backbone(cfg: MyModelConfig, L: int, N: int, T: int, in_extra: int) -> nn.Module:
    typ = cfg.backbone_type.lower()
    if typ in ("gru", "lstm"):
        return RNNBackbone(
            rnn_type=typ,
            L=L, N=N, T=T,
            in_extra=in_extra,
            D=int(cfg.backbone_hidden_size),
            layers=int(cfg.backbone_layers),
            dropout=float(cfg.backbone_dropout),
            use_ts=bool(cfg.use_input_timestamps),
        )
    elif typ == "transformer":
        return TransformerBackbone(
            L=L, N=N, T=T,
            in_extra=in_extra,
            D=int(cfg.backbone_hidden_size),
            layers=int(cfg.backbone_layers),
            nhead=int(cfg.transformer_nhead),
            ffn_ratio=float(cfg.transformer_ffn_ratio),
            dropout=float(cfg.backbone_dropout),
            norm_first=bool(cfg.transformer_norm_first),
            use_pe=bool(cfg.transformer_use_positional_encoding),
            use_ts=bool(cfg.use_input_timestamps),
        )
    else:
        raise ValueError(f"Unknown backbone_type: {cfg.backbone_type}")


# ============================================================
# Branch: Spatial (low-rank, avoids N×N)
# ============================================================

class SpatialLowRank(nn.Module):
    def __init__(self, N: int, D: int, r: int, T: int, S: int, hidden: int, dropout: float, alpha: float, reg_orth: float):
        super().__init__()
        self.N, self.D, self.r = int(N), int(D), int(r)
        self.T, self.S = int(T), int(S)
        self.alpha = float(alpha)
        self.reg_orth = float(reg_orth)

        self.B = nn.Parameter(torch.empty(self.N, self.r))
        nn.init.xavier_uniform_(self.B)

        in_dim = self.r + self.T + self.S
        self.scale_mlp = nn.Sequential(
            nn.Linear(in_dim, int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), self.r),
        )

    def forward(self, H: torch.Tensor, ts_out: Optional[torch.Tensor], step_emb: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        H: [B,N,D]
        ts_out: [B,O,T] or None
        step_emb: [B,O,S] or None
        return: H_spatial [B,O,N,D], info
        """
        Bsz, N, D = H.shape
        Bmat = self.B
        U = torch.einsum("bnd,nr->brd", H, Bmat)  # [B,r,D]
        ctx = torch.sqrt(torch.mean(U ** 2, dim=-1) + 1e-6)  # [B,r]

        # build features for s(b,o)
        O = 1
        parts = [ctx.unsqueeze(1)]
        if ts_out is not None:
            O = ts_out.size(1)
            parts.append(ts_out.float())
        if step_emb is not None:
            O = step_emb.size(1)
            parts.append(step_emb.float())
        feat = torch.cat([p.expand(-1, O, -1) if p.dim() == 3 and p.size(1) == 1 else p for p in parts], dim=-1)

        s = torch.nn.functional.softplus(self.scale_mlp(feat)) + 1e-6  # [B,O,r], >=0

        SU = s.unsqueeze(-1) * U.unsqueeze(1)               # [B,O,r,D]
        M = torch.einsum("bord,nr->bond", SU, Bmat)         # [B,O,N,D]

        H_base = H.unsqueeze(1).expand(-1, O, -1, -1)
        H_spatial = H_base + self.alpha * (M - H_base)

        info = {"graph_basis": Bmat, "graph_scales": s}
        return H_spatial, info

    def orth_reg(self) -> torch.Tensor:
        if self.reg_orth <= 0:
            return self.B.new_zeros(())
        BtB = (self.B.t() @ self.B) / float(self.N)  # [r,r]
        I = torch.eye(self.r, device=self.B.device, dtype=self.B.dtype)
        return self.reg_orth * torch.mean((BtB - I) ** 2)


# ============================================================
# Branch: Time (NEW) Spectral-Token Attention
# ============================================================

class SpectralTokenTimeAttention(nn.Module):
    """
    Novel time attention for traffic:
      - Build M periodic tokens = K_tod daily harmonics + K_dow weekly harmonics.
      - Each token m has sin/cos value at the forecast timestamp (phase).
      - Query comes from node state (and step embedding).
      - Attention weights over tokens are interpretable: which periodicities matter now.

    Output is a feature residual delta_time [B,O,N,D] added to H_base.
    """
    def __init__(self, D: int, S: int, K_tod: int, K_dow: int, attn_dim: int, alpha: float, gate_bound: float):
        super().__init__()
        self.D, self.S = int(D), int(S)
        self.K_tod, self.K_dow = int(K_tod), int(K_dow)
        self.M = self.K_tod + self.K_dow
        self.attn_dim = int(attn_dim)
        self.alpha = float(alpha)
        self.gate_bound = float(gate_bound)

        # token ids: 0..M-1, includes both tod and dow harmonics
        self.key_emb = nn.Embedding(self.M, self.attn_dim)
        self.val_emb = nn.Embedding(self.M, self.D)

        self.k_proj = nn.Linear(2, self.attn_dim, bias=False)
        self.g_proj = nn.Linear(2, 1, bias=True)

        q_in = self.D + self.S
        self.q_proj = nn.Linear(q_in, self.attn_dim, bias=True)

    def forward(self, H: torch.Tensor, ts_out: torch.Tensor, step_emb: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        H: [B,N,D]
        ts_out: [B,O,T>=2] uses first two dims: [tod, dow]
        step_emb: [B,O,S] (or None if S=0)
        """
        Bsz, N, D = H.shape
        O = ts_out.size(1)
        tod = ts_out[..., 0]
        dow = ts_out[..., 1]

        # build tokens: [B,O,M,2]
        toks = []
        for k in range(1, self.K_tod + 1):
            toks.append(_fourier_sincos(tod, k))
        for k in range(1, self.K_dow + 1):
            toks.append(_fourier_sincos(dow, k))
        tok = torch.stack(toks, dim=2) if toks else tod.new_zeros((Bsz, O, 0, 2))  # [B,O,M,2]

        token_ids = torch.arange(self.M, device=H.device, dtype=torch.long)  # [M]
        key = self.k_proj(tok) + self.key_emb(token_ids).view(1, 1, self.M, self.attn_dim)  # [B,O,M,A]
        gate = torch.tanh(self.g_proj(tok)) * self.gate_bound                                  # [B,O,M,1]
        val = gate * self.val_emb(token_ids).view(1, 1, self.M, self.D)                        # [B,O,M,D]

        H_rep = H.unsqueeze(1).expand(-1, O, -1, -1)  # [B,O,N,D]
        if self.S > 0 and step_emb is not None:
            step_rep = step_emb.unsqueeze(2).expand(-1, -1, N, -1)  # [B,O,N,S]
            q_in = torch.cat([H_rep, step_rep], dim=-1)
        else:
            q_in = torch.cat([H_rep, H_rep.new_zeros((Bsz, O, N, 0))], dim=-1)

        q = self.q_proj(q_in)  # [B,O,N,A]

        # attn logits: [B,O,N,M]
        logits = torch.einsum("bona,boma->bonm", q, key) / math.sqrt(float(self.attn_dim))
        attn = torch.softmax(logits, dim=-1)

        delta = torch.einsum("bonm,bomd->bond", attn, val)  # [B,O,N,D]
        H_time = H_rep + self.alpha * delta

        info = {"time_attn": attn, "time_gate": gate.squeeze(-1)}
        return H_time, info


# ============================================================
# Convex fusion: base + spatial + time
# ============================================================

class ConvexFusion3(nn.Module):
    """
    weights:
      us = softplus(raw_s), ut = softplus(raw_t)
      ws = us/(1+us+ut), wt = ut/(1+us+ut), w0 = 1/(1+us+ut)
    """
    def __init__(self, raw_s_init: float, raw_t_init: float, learnable: bool = True):
        super().__init__()
        rs = torch.tensor(float(raw_s_init))
        rt = torch.tensor(float(raw_t_init))
        if learnable:
            self.raw_s = nn.Parameter(rs)
            self.raw_t = nn.Parameter(rt)
        else:
            self.register_buffer("raw_s", rs, persistent=True)
            self.register_buffer("raw_t", rt, persistent=True)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        us = torch.nn.functional.softplus(self.raw_s)
        ut = torch.nn.functional.softplus(self.raw_t)
        denom = 1.0 + us + ut
        w0 = 1.0 / denom
        ws = us / denom
        wt = ut / denom
        return w0, ws, wt


# ============================================================
# Decoder blocks
# ============================================================

class DecoderMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        if hidden <= 0:
            self.net = nn.Linear(in_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, int(hidden)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden), 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ParamHead(nn.Module):
    """Generic parameter head: outputs K scalars."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        if hidden <= 0:
            self.net = nn.Linear(in_dim, out_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, int(hidden)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden), out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Main model
# ============================================================

class MyModel(nn.Module):
    def __init__(self, cfg: MyModelConfig):
        super().__init__()
        self.cfg = cfg

        # fixed shapes
        self.L = int(cfg.input_len)
        self.O = int(cfg.output_len)
        self.N = int(cfg.num_features)
        self.T = int(cfg.num_timestamps)
        self.D = int(cfg.backbone_hidden_size)

        self.last_value_centering = bool(cfg.last_value_centering)

        # embeddings
        self.node_emb_dim = int(cfg.node_emb_dim)
        self.step_emb_dim = int(cfg.step_emb_dim)
        self.drop = nn.Dropout(float(cfg.dropout))

        self.node_emb = nn.Embedding(self.N, self.node_emb_dim) if self.node_emb_dim > 0 else None
        self.step_emb = nn.Embedding(self.O, self.step_emb_dim) if self.step_emb_dim > 0 else None

        # backbone input extra dims = node_emb_dim (we always inject node id into backbone when enabled)
        self.backbone = build_backbone(cfg, L=self.L, N=self.N, T=self.T, in_extra=self.node_emb_dim)

        # branches
        self.enable_spatial = bool(cfg.enable_spatial)
        self.spatial = None
        if self.enable_spatial:
            self.spatial = SpatialLowRank(
                N=self.N, D=self.D, r=int(cfg.spatial_rank),
                T=self.T if bool(cfg.spatial_use_output_timestamps) else 0,
                S=self.step_emb_dim,
                hidden=int(cfg.spatial_scale_hidden),
                dropout=float(cfg.spatial_scale_dropout),
                alpha=float(cfg.spatial_alpha),
                reg_orth=float(cfg.reg_spatial_orth),
            )

        self.enable_time = bool(cfg.enable_time)
        self.time = None
        if self.enable_time:
            self.time = SpectralTokenTimeAttention(
                D=self.D,
                S=self.step_emb_dim,
                K_tod=int(cfg.time_tod_harmonics),
                K_dow=int(cfg.time_dow_harmonics),
                attn_dim=int(cfg.time_attn_dim),
                alpha=float(cfg.time_alpha),
                gate_bound=float(cfg.time_gate_bound),
            )

        # fusion
        self.fusion = ConvexFusion3(
            raw_s_init=float(cfg.fusion_raw_spatial_init),
            raw_t_init=float(cfg.fusion_raw_time_init),
            learnable=bool(cfg.fusion_learnable),
        )

        # decoder input dim
        dec_in = self.D
        if self.step_emb_dim > 0:
            dec_in += self.step_emb_dim
        if self.node_emb_dim > 0:
            dec_in += self.node_emb_dim
        self.decoder_use_ts_out = bool(cfg.decoder_use_output_timestamps)
        if self.decoder_use_ts_out:
            dec_in += self.T

        self.mu_head = DecoderMLP(in_dim=dec_in, hidden=int(cfg.decoder_mlp_hidden), dropout=float(cfg.dropout))

        # distribution parameter heads (share same dec_in)
        self.likelihood = cfg.likelihood.lower()
        self.min_scale = float(cfg.min_scale)

        self.scale_head = None
        self.quantile_head = None
        if self.likelihood in ("gaussian", "studentt"):
            self.scale_head = ParamHead(dec_in, hidden=int(cfg.decoder_mlp_hidden), out_dim=1, dropout=float(cfg.dropout))
        elif self.likelihood == "quantile":
            self.quantile_levels = list(cfg.quantiles)
            self.quantile_head = ParamHead(dec_in, hidden=int(cfg.decoder_mlp_hidden), out_dim=len(self.quantile_levels), dropout=float(cfg.dropout))

        # student-t df
        self.studentt_df_mode = cfg.studentt_df_mode
        self.studentt_df_min = float(cfg.studentt_df_min)
        if self.likelihood == "studentt":
            if self.studentt_df_mode == "learned_global":
                df_init = float(cfg.studentt_df_init)
                self.df_raw = nn.Parameter(torch.tensor(df_init))
            else:
                self.register_buffer("df_raw", torch.tensor(float(cfg.studentt_df_init)), persistent=False)

        # linear skip (delta space)
        self.enable_linear_skip = bool(cfg.enable_linear_skip)
        if self.enable_linear_skip:
            self.linear_skip = nn.Linear(self.L, self.O, bias=True)
        else:
            self.linear_skip = None

        # loss
        self.point_loss = cfg.point_loss.lower()
        self.huber_delta = float(cfg.huber_delta)
        self.lambda_point = float(cfg.lambda_point)
        self.lambda_nll = float(cfg.lambda_nll)
        self.compute_loss_in_forward = bool(cfg.compute_loss_in_forward)

        # outputs
        self.return_interpretation = bool(cfg.return_interpretation)
        self.return_components = bool(cfg.return_components)

        # cached indices
        self.register_buffer("_node_idx", torch.arange(self.N, dtype=torch.long), persistent=False)
        self.register_buffer("_step_idx", torch.arange(self.O, dtype=torch.long), persistent=False)

    def _select_tap(self, H_layers: torch.Tensor) -> torch.Tensor:
        # H_layers: [Llayers,B,N,D]
        Llayers = H_layers.size(0)
        k = int(self.cfg.backbone_tap_layer)
        if k < 0:
            k = Llayers + k
        k = max(0, min(Llayers - 1, k))
        return H_layers[k]  # [B,N,D]

    def _build_decoder_input(self, F: torch.Tensor, ts_out: Optional[torch.Tensor]) -> torch.Tensor:
        # F: [B,O,N,D]
        Bsz, O, N, D = F.shape

        parts = [F]

        if self.step_emb is not None:
            se = self.drop(self.step_emb(self._step_idx))  # [O,S]
            se = se.unsqueeze(0).unsqueeze(2).expand(Bsz, O, N, -1)
            parts.append(se)

        if self.node_emb is not None:
            ne = self.drop(self.node_emb(self._node_idx))  # [N,E]
            ne = ne.unsqueeze(0).unsqueeze(1).expand(Bsz, O, N, -1)
            parts.append(ne)

        if self.decoder_use_ts_out:
            if ts_out is None:
                raise ValueError("decoder_use_output_timestamps=True but targets_timestamps is None.")
            ts = ts_out.float().unsqueeze(2).expand(Bsz, O, N, -1)
            parts.append(ts)

        return torch.cat(parts, dim=-1)

    def _compute_point_loss(self, pred: torch.Tensor, true: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.point_loss == "mae":
            return masked_mae(pred, true, mask)
        elif self.point_loss == "mse":
            return masked_mse(pred, true, mask)
        elif self.point_loss == "huber":
            return masked_huber(pred, true, mask, delta=self.huber_delta)
        else:
            raise ValueError(f"Unknown point_loss: {self.point_loss}")

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        inputs_timestamps: Optional[torch.Tensor] = None,
        targets_timestamps: Optional[torch.Tensor] = None,
        targets_mask: Optional[torch.Tensor] = None,
        train: bool = False,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> Dict:
        """
        Required return:
          - prediction: [B,O,N]
          - loss: scalar, if targets provided and compute_loss_in_forward=True
        """
        Bsz, L, N = inputs.shape
        if L != self.L or N != self.N:
            raise ValueError(f"inputs must be [B,{self.L},{self.N}], got {tuple(inputs.shape)}")

        # timestamps requirement
        if (self.enable_time or (self.enable_spatial and self.cfg.spatial_use_output_timestamps) or self.decoder_use_ts_out) and (targets_timestamps is None):
            raise ValueError("targets_timestamps is required by current configuration.")

        # center (delta)
        if self.last_value_centering:
            last = inputs[:, -1, :]           # [B,N]
            x0 = inputs - last.unsqueeze(1)   # [B,L,N]
        else:
            last = None
            x0 = inputs

        # node embedding for backbone input
        node_feat = None
        if self.node_emb is not None:
            node_feat = self.drop(self.node_emb(self._node_idx))  # [N,E]

        # backbone
        H_layers = self.backbone(x0, inputs_timestamps, node_feat)   # [Llayers,B,N,D]
        H = self._select_tap(H_layers)                               # [B,N,D]

        # step embedding for branches
        step_feat = None
        if self.step_emb is not None:
            se = self.drop(self.step_emb(self._step_idx))            # [O,S]
            step_feat = se.unsqueeze(0).expand(Bsz, -1, -1)          # [B,O,S]

        H_base = H.unsqueeze(1).expand(-1, self.O, -1, -1)           # [B,O,N,D]

        # spatial branch
        spatial_info: Dict[str, torch.Tensor] = {}
        H_spatial = H_base
        if self.spatial is not None:
            ts_for_spatial = targets_timestamps if self.cfg.spatial_use_output_timestamps else None
            H_spatial, spatial_info = self.spatial(H, ts_for_spatial, step_feat)

        # time branch
        time_info: Dict[str, torch.Tensor] = {}
        H_time = H_base
        if self.time is not None:
            H_time, time_info = self.time(H, targets_timestamps, step_feat)

        # convex fusion
        w0, ws, wt = self.fusion()
        F = w0 * H_base + ws * H_spatial + wt * H_time               # [B,O,N,D]

        # decode
        dec_in = self._build_decoder_input(F, targets_timestamps if self.decoder_use_ts_out else None)
        mu_delta = self.mu_head(dec_in)                              # [B,O,N]

        # linear skip in delta space
        if self.linear_skip is not None:
            lin = self.linear_skip(x0.permute(0, 2, 1))              # [B,N,O]
            mu_delta = mu_delta + lin.permute(0, 2, 1)

        # restore level
        mu = mu_delta
        if last is not None:
            mu = mu + last.unsqueeze(1)

        out: Dict = {"prediction": mu}

        # distribution params
        dist_params: Dict[str, torch.Tensor] = {}
        if self.likelihood in ("gaussian", "studentt"):
            raw_scale = self.scale_head(dec_in).squeeze(-1)          # [B,O,N]
            scale = torch.nn.functional.softplus(raw_scale) + self.min_scale
            dist_params["scale"] = scale
            if self.likelihood == "studentt":
                df = torch.nn.functional.softplus(self.df_raw) + self.studentt_df_min
                dist_params["df"] = df
            out["dist_name"] = self.likelihood
            out["dist_params"] = dist_params
        elif self.likelihood == "quantile":
            q_raw = self.quantile_head(dec_in)                       # [B,O,N,Q]
            out["dist_name"] = "quantile"
            out["dist_params"] = {"quantiles": q_raw, "levels": torch.tensor(self.quantile_levels, device=mu.device)}
        else:
            out["dist_name"] = "none"

        # optional components / interpretation
        if self.return_components:
            out["components"] = {
                "w_base": w0.detach(),
                "w_spatial": ws.detach(),
                "w_time": wt.detach(),
            }
        if self.return_interpretation:
            # for paper: show spectral token attention weights and low-rank graph modes
            out["interpretation"] = {
                "fusion_weights": torch.stack([w0.detach(), ws.detach(), wt.detach()]),
                **{k: v.detach() for k, v in spatial_info.items()},
                # to avoid huge memory, we provide mean over nodes for time attention
                "time_attn_mean_over_nodes": (time_info["time_attn"].detach().mean(dim=2) if "time_attn" in time_info else None),
                "time_gate": (time_info["time_gate"].detach() if "time_gate" in time_info else None),
            }

        # loss inside forward (runner compatibility)
        if self.compute_loss_in_forward and (targets is not None):
            # point loss
            loss_point = self._compute_point_loss(mu, targets, targets_mask)
            loss = self.lambda_point * loss_point

            # nll
            loss_nll = mu.new_zeros(())
            if self.likelihood == "gaussian":
                nll = gaussian_nll(targets, mu, dist_params["scale"])
                loss_nll = masked_mean(nll, targets_mask)
                loss = loss + self.lambda_nll * loss_nll
            elif self.likelihood == "studentt":
                nll = studentt_nll(targets, mu, dist_params["scale"], dist_params["df"])
                loss_nll = masked_mean(nll, targets_mask)
                loss = loss + self.lambda_nll * loss_nll
            elif self.likelihood == "quantile":
                q_raw = out["dist_params"]["quantiles"]  # [B,O,N,Q]
                levels = self.quantile_levels
                # sum pinball losses for all quantiles
                q_losses = []
                for i, q in enumerate(levels):
                    q_losses.append(quantile_loss(targets, q_raw[..., i], float(q)))
                q_loss = torch.stack(q_losses, dim=-1).mean(dim=-1)  # mean over Q
                loss_nll = masked_mean(q_loss, targets_mask)
                loss = loss + self.lambda_nll * loss_nll

            # spatial regularization (orth)
            loss_reg = mu.new_zeros(())
            if self.spatial is not None:
                loss_reg = loss_reg + self.spatial.orth_reg()
            loss = loss + loss_reg

            out["loss"] = loss
            out["loss_point"] = loss_point.detach()
            out["loss_nll"] = loss_nll.detach()
            out["loss_reg"] = loss_reg.detach()

        return out
