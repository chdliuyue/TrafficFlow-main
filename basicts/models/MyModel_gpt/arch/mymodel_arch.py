
import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from ..config.mymodel_config import MyModelConfig
from ....metrics.myloss import RegWeights, compute_total_loss


# =========================================================
# Backbone: stacked LSTM/GRU/Transformer over time (shared across nodes)
# =========================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding (batch_first=True)."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0).to(x.dtype)


class NodeTemporalBackbone(nn.Module):
    """
    Shared temporal backbone across nodes.

    Inputs:
      values:    [B, L, N]
      ts_in:     [B, L, T] (optional, already normalized to [0,1])
      node_emb:  [N, E]    (optional; used iff cfg.node_emb_in_backbone=True and E>0)

    Output:
      H: [B, N, D]  (node-wise trunk representation)
    """

    def __init__(self, cfg: MyModelConfig):
        super().__init__()
        self.cfg = cfg
        self.L = int(cfg.input_len)
        self.N = int(cfg.num_features)
        self.T = int(cfg.num_timestamps)

        self.use_in_ts = bool(cfg.use_input_timestamps)
        self.node_in = bool(getattr(cfg, "node_emb_in_backbone", True)) and int(getattr(cfg, "node_emb_dim", 0)) > 0
        self.node_emb_dim = int(getattr(cfg, "node_emb_dim", 0))

        in_dim = 1 + (self.T if self.use_in_ts else 0) + (self.node_emb_dim if self.node_in else 0)

        btype = str(cfg.backbone_type).lower()
        self.backbone_type = btype
        D = int(cfg.backbone_hidden_size)
        layers = int(cfg.backbone_layers)
        drop = float(cfg.backbone_dropout)

        if btype in {"lstm", "gru"}:
            rnn_cls = nn.LSTM if btype == "lstm" else nn.GRU
            self.rnn = rnn_cls(
                input_size=in_dim,
                hidden_size=D,
                num_layers=layers,
                dropout=drop if layers > 1 else 0.0,
                batch_first=True,
            )
            self.input_proj = None
            self.pe = None
            self.transformer = None

        elif btype == "transformer":
            self.rnn = None
            self.input_proj = nn.Linear(in_dim, D)
            self.pe = SinusoidalPositionalEncoding(d_model=D, max_len=self.L) if bool(cfg.transformer_use_positional_encoding) else nn.Identity()

            nhead = int(cfg.transformer_nhead)
            ffn_dim = int(D * float(cfg.transformer_ffn_ratio))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=D,
                nhead=nhead,
                dim_feedforward=ffn_dim,
                dropout=drop,
                activation="gelu",
                batch_first=True,
                norm_first=bool(cfg.transformer_norm_first),
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)

        else:
            raise ValueError(f"Unknown backbone_type: {btype} (expected lstm|gru|transformer).")

        self.out_dim = D

    def forward(self, values: torch.Tensor, ts_in: Optional[torch.Tensor], node_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(f"Expected values [B,L,N], got {tuple(values.shape)}")
        B, L, N = values.shape
        if L != self.L or N != self.N:
            raise ValueError(f"Input shape mismatch: got L={L},N={N}, expected L={self.L},N={self.N}")

        # per-node sequence: [B,N,L,1]
        v = values.permute(0, 2, 1).unsqueeze(-1)

        feats = [v]

        if self.use_in_ts:
            if ts_in is None:
                raise ValueError("use_input_timestamps=True but inputs_timestamps is None.")
            if ts_in.shape[:2] != (B, L):
                raise ValueError(f"inputs_timestamps should be [B,L,T], got {tuple(ts_in.shape)}")
            ts = ts_in.unsqueeze(1).expand(-1, N, -1, -1).float()
            feats.append(ts)

        if self.node_in:
            if node_emb is None:
                raise ValueError("node_emb_in_backbone=True but node_emb is None.")
            if node_emb.shape[0] != N or node_emb.shape[1] != self.node_emb_dim:
                raise ValueError(f"node_emb should be [N,{self.node_emb_dim}], got {tuple(node_emb.shape)}")
            ne = node_emb.unsqueeze(0).unsqueeze(2).expand(B, N, L, -1).to(v.dtype)
            feats.append(ne)

        x = torch.cat(feats, dim=-1)  # [B,N,L,in_dim]
        x = x.reshape(B * N, L, x.size(-1))  # [B*N,L,in_dim]

        if self.backbone_type in {"lstm", "gru"}:
            _, h = self.rnn(x)
            if self.backbone_type == "lstm":
                h_last = h[0][-1]  # [B*N,D]
            else:
                h_last = h[-1]
            return h_last.reshape(B, N, -1)

        z = self.input_proj(x)
        z = self.pe(z)
        z = self.transformer(z)  # [B*N,L,D]
        h_last = z[:, -1, :]
        return h_last.reshape(B, N, -1)


# =========================================================
# Module B) Time interpretability: Fourier basis (+ optional step FiLM)
# =========================================================

class FourierTimeEffect(nn.Module):
    r"""
    Time effect with a fixed Fourier basis and state-conditioned coefficients:

      b_time(b,o,n) = <c_tod(b,n;H), phi_tod(tod_{b,o})> + <c_dow(b,n;H), phi_dow(dow_{b,o})>

    Optional step embedding modulation (FiLM) upgrades it to:
      c_tod(b,o,n) = c_state(b,n) ⊙ (1 + s_step(o)) + b_step(o)

    This keeps interpretability:
      - phi_* are explicit sine/cosine bases (periodic, mean-zero).
      - coefficients are decomposable into (state part) and (horizon part).
    """

    def __init__(
        self,
        feat_dim: int,
        node_emb_dim: int,
        step_emb_dim: int,
        K_tod: int,
        K_dow: int,
        coef_hidden: int = 0,
        dropout: float = 0.0,
        use_step_film: bool = True,
    ):
        super().__init__()
        self.D = int(feat_dim)
        self.E = int(node_emb_dim)
        self.S = int(step_emb_dim)
        self.K_tod = int(K_tod)
        self.K_dow = int(K_dow)
        self.use_step_film = bool(use_step_film) and self.S > 0

        self.in_dim = self.D + (self.E if self.E > 0 else 0)

        P_tod = 2 * self.K_tod
        P_dow = 2 * self.K_dow
        self.P_tod = P_tod
        self.P_dow = P_dow

        def make_coef_net(P: int):
            if P == 0:
                return None
            if int(coef_hidden) <= 0:
                return nn.Linear(self.in_dim, P, bias=True)
            return nn.Sequential(
                nn.Linear(self.in_dim, int(coef_hidden)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(coef_hidden), P),
            )

        self.coef_tod = make_coef_net(P_tod)
        self.coef_dow = make_coef_net(P_dow)

        # step FiLM: scale/shift per horizon
        if self.use_step_film and P_tod > 0:
            self.step_scale_tod = nn.Linear(self.S, P_tod, bias=True)
            self.step_shift_tod = nn.Linear(self.S, P_tod, bias=True)
        else:
            self.step_scale_tod = None
            self.step_shift_tod = None

        if self.use_step_film and P_dow > 0:
            self.step_scale_dow = nn.Linear(self.S, P_dow, bias=True)
            self.step_shift_dow = nn.Linear(self.S, P_dow, bias=True)
        else:
            self.step_scale_dow = None
            self.step_shift_dow = None

    @staticmethod
    def _fourier_basis(x: torch.Tensor, K: int) -> torch.Tensor:
        """
        x: [B,O] in [0,1]
        returns: [B,O,2K] = [sin(2πkx), cos(2πkx)]_{k=1..K}
        """
        if K <= 0:
            return x.new_zeros((*x.shape, 0))
        ks = torch.arange(1, K + 1, device=x.device, dtype=x.dtype).view(1, 1, K)  # [1,1,K]
        ang = 2.0 * math.pi * x.unsqueeze(-1) * ks                                  # [B,O,K]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)                  # [B,O,2K]

    def forward(
        self,
        H: torch.Tensor,                        # [B,N,D]
        ts_out: torch.Tensor,                   # [B,O,T], T>=2
        node_emb: Optional[torch.Tensor] = None,  # [N,E]
        step_emb: Optional[torch.Tensor] = None,  # [O,S]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if ts_out is None:
            raise ValueError("targets_timestamps is required for time effect.")
        if H.ndim != 3:
            raise ValueError(f"Expected H [B,N,D], got {tuple(H.shape)}")
        B, N, D = H.shape
        Bt, O, T = ts_out.shape
        if Bt != B:
            raise ValueError("Batch mismatch between H and targets_timestamps")
        if T < 2:
            raise ValueError("FourierTimeEffect expects at least 2 timestamp dims: [tod_norm, dow_norm].")

        tod = ts_out[..., 0].float()  # [B,O]
        dow = ts_out[..., 1].float()  # [B,O]

        basis_tod = self._fourier_basis(tod, self.K_tod)  # [B,O,P_tod]
        basis_dow = self._fourier_basis(dow, self.K_dow)  # [B,O,P_dow]

        # build H_aug = [H, node_emb]
        H_aug = H
        if self.E > 0:
            if node_emb is None:
                raise ValueError("node_emb_dim>0 but node_emb is None for time effect.")
            if node_emb.shape[0] != N or node_emb.shape[1] != self.E:
                raise ValueError(f"node_emb should be [N,{self.E}], got {tuple(node_emb.shape)}")
            ne = node_emb.unsqueeze(0).expand(B, -1, -1).to(H.dtype)
            H_aug = torch.cat([H, ne], dim=-1)

        time_tod = H.new_zeros(B, O, N)
        time_dow = H.new_zeros(B, O, N)

        coef_state_tod = None
        coef_state_dow = None

        if self.coef_tod is not None and basis_tod.numel() > 0:
            coef_state_tod = self.coef_tod(H_aug)  # [B,N,P_tod]
            if self.use_step_film and (step_emb is not None) and (self.step_scale_tod is not None):
                if step_emb.shape[0] != O or step_emb.shape[1] != self.S:
                    raise ValueError(f"step_emb should be [O,{self.S}], got {tuple(step_emb.shape)}")
                ss = self.step_scale_tod(step_emb).to(H.dtype)  # [O,P_tod]
                bb = self.step_shift_tod(step_emb).to(H.dtype)  # [O,P_tod]
                coef_full = coef_state_tod.unsqueeze(1) * (1.0 + ss.unsqueeze(0).unsqueeze(2)) + bb.unsqueeze(0).unsqueeze(2)  # [B,O,N,P]
                time_tod = torch.einsum("bop,bonp->bon", basis_tod, coef_full)
            else:
                time_tod = torch.einsum("bop,bnp->bon", basis_tod, coef_state_tod)

        if self.coef_dow is not None and basis_dow.numel() > 0:
            coef_state_dow = self.coef_dow(H_aug)  # [B,N,P_dow]
            if self.use_step_film and (step_emb is not None) and (self.step_scale_dow is not None):
                if step_emb.shape[0] != O or step_emb.shape[1] != self.S:
                    raise ValueError(f"step_emb should be [O,{self.S}], got {tuple(step_emb.shape)}")
                ss = self.step_scale_dow(step_emb).to(H.dtype)  # [O,P_dow]
                bb = self.step_shift_dow(step_emb).to(H.dtype)  # [O,P_dow]
                coef_full = coef_state_dow.unsqueeze(1) * (1.0 + ss.unsqueeze(0).unsqueeze(2)) + bb.unsqueeze(0).unsqueeze(2)
                time_dow = torch.einsum("bop,bonp->bon", basis_dow, coef_full)
            else:
                time_dow = torch.einsum("bop,bnp->bon", basis_dow, coef_state_dow)

        time_total = time_tod + time_dow

        info = {
            "time_tod": time_tod,
            "time_dow": time_dow,
            "time_total": time_total,
            "time_coef_tod_state": coef_state_tod,  # [B,N,P_tod] or None
            "time_coef_dow_state": coef_state_dow,  # [B,N,P_dow] or None
            "time_basis_tod": basis_tod,            # [B,O,P_tod]
            "time_basis_dow": basis_dow,            # [B,O,P_dow]
        }
        return time_total, info


# =========================================================
# Module A) Spatial interpretability: low-rank PSD dynamic kernel on H (no N×N)
# =========================================================

class LowRankDynamicKernelOnFeatures(nn.Module):
    r"""
    Adjacency-free dynamic interaction on features H via a low-rank operator.

    For each horizon o:
        A(b,o) = B diag(s(b,o)) B^T,   s(b,o) >= 0  =>  A(b,o) PSD, rank<=r.

    We DO NOT materialize A (N×N). Compute:
        M = A H = B ( s ⊙ (B^T H) )     with O(N*r*D).

    Output (module-internal convex update):
        H_graph = H_base + alpha*(M - H_base), alpha in [0,1].

    NOTE on normalization:
      - If graph_normalize=True, the current implementation uses a degree-like normalization based on B.sum(dim=0),
        which is meaningful/stable mainly when basis entries are nonnegative.
      - For signed bases (graph_nonnegative_basis=False), we strongly recommend graph_normalize=False (as your exp-8 showed).
    """

    def __init__(
        self,
        num_features: int,
        feat_dim: int,
        rank: int,
        num_timestamps: int,
        use_output_timestamps: bool,
        hidden_size: int,
        dropout: float,
        nonnegative_basis: bool,
        normalize: bool,
        alpha: float,
        step_emb_dim: int = 0,
        use_step_embedding: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.N = int(num_features)
        self.D = int(feat_dim)
        self.r = int(rank)
        self.T = int(num_timestamps)
        self.use_out_ts = bool(use_output_timestamps)
        self.use_step = bool(use_step_embedding) and int(step_emb_dim) > 0
        self.S = int(step_emb_dim) if self.use_step else 0

        self.nonneg = bool(nonnegative_basis)
        self.normalize = bool(normalize)
        self.eps = float(eps)

        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

        self._basis = nn.Parameter(torch.empty(self.N, self.r))
        nn.init.xavier_uniform_(self._basis)

        in_dim = self.r + (self.T if self.use_out_ts else 0) + (self.S if self.use_step else 0)
        self.scale_mlp = nn.Sequential(
            nn.Linear(in_dim, int(hidden_size)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), self.r),
        )

    def basis(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._basis) if self.nonneg else self._basis

    def forward(
        self,
        H: torch.Tensor,                               # [B,N,D]
        ts_out: Optional[torch.Tensor],                # [B,O,T]
        O: int,
        step_emb: Optional[torch.Tensor] = None,       # [O,S]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if H.ndim != 3:
            raise ValueError(f"Expected H [B,N,D], got {tuple(H.shape)}")
        Bsz, N, D = H.shape
        if N != self.N or D != self.D:
            raise ValueError(f"H shape mismatch: got {tuple(H.shape)}, expected [B,{self.N},{self.D}]")

        Bmat = self.basis()                                   # [N,r]
        U = torch.einsum("bnd,nr->brd", H, Bmat)              # [B,r,D]
        ctx = torch.sqrt(torch.mean(U ** 2, dim=-1) + self.eps)  # [B,r]

        feat_parts = []
        if self.use_out_ts:
            if ts_out is None:
                raise ValueError("graph_use_output_timestamps=True but targets_timestamps is None.")
            if ts_out.shape[0] != Bsz or ts_out.shape[1] != O:
                raise ValueError(f"targets_timestamps should be [B,O,T], got {tuple(ts_out.shape)}")
            feat_parts.append(ts_out.float())  # [B,O,T]

        feat_parts.append(ctx.unsqueeze(1).expand(-1, O, -1))  # [B,O,r]

        if self.use_step:
            if step_emb is None:
                raise ValueError("use_step_embedding=True but step_emb is None.")
            if step_emb.shape[0] != O or step_emb.shape[1] != self.S:
                raise ValueError(f"step_emb should be [O,{self.S}], got {tuple(step_emb.shape)}")
            se = step_emb.unsqueeze(0).expand(Bsz, -1, -1).to(ctx.dtype)  # [B,O,S]
            feat_parts.append(se)

        feat = torch.cat(feat_parts, dim=-1)  # [B,O, ...]
        s_raw = self.scale_mlp(feat)          # [B,O,r]
        s = torch.nn.functional.softplus(s_raw)  # >=0

        # M = B ( s ⊙ (B^T H) )
        Uexp = U.unsqueeze(1)                    # [B,1,r,D]
        X = s.unsqueeze(-1) * Uexp               # [B,O,r,D]
        M = torch.einsum("bord,nr->bond", X, Bmat)  # [B,O,N,D]

        if self.normalize:
            b_sum = Bmat.sum(dim=0)                        # [r]
            deg = torch.einsum("bor,nr->bon", s * b_sum.view(1, 1, -1), Bmat)  # [B,O,N]
            M = M / (deg.unsqueeze(-1) + self.eps)

        H_base = H.unsqueeze(1).expand(-1, O, -1, -1)
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        H_graph = H_base + alpha * (M - H_base)

        info = {
            "graph_basis": Bmat,      # [N,r]
            "graph_scales": s,        # [B,O,r]
            "graph_mode_proj": U,     # [B,r,D]
            "graph_alpha": alpha.detach(),
        }
        return H_graph, info


# =========================================================
# Fusion: convex weights (base + graph), with configurable granularity
# =========================================================

class ConvexGraphFusion(nn.Module):
    """
    Convex weights for combining base and graph features.

    Output:
      dict {"w_base": ..., "w_graph": ...}

    Shapes by fusion_mode:
      - global:           []          (scalar)
      - per_horizon:      [O]
      - per_node:         [N]
      - per_node_horizon: [O, N]      (factorized: raw0 + raw_step[o] + raw_node[n])
    """

    def __init__(
        self,
        learnable: bool,
        raw_init: float,
        mode: str,
        num_nodes: int,
        num_horizons: int,
        w_min: float = 0.0,
        w_max: float = 1.0,
    ):
        super().__init__()
        self.learnable = bool(learnable)
        self.mode = str(mode).lower()
        self.N = int(num_nodes)
        self.O = int(num_horizons)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        if not (0.0 <= self.w_min <= self.w_max <= 1.0):
            raise ValueError(f"fusion_w_min/max must satisfy 0<=min<=max<=1, got {self.w_min},{self.w_max}")

        def make_param(shape, init):
            t = torch.full(shape, float(init))
            return nn.Parameter(t) if self.learnable else t

        # global logit
        self.raw0 = make_param((), raw_init)

        if self.mode == "global":
            self.raw_step = None
            self.raw_node = None
        elif self.mode == "per_horizon":
            self.raw_step = make_param((self.O,), raw_init)
            self.raw_node = None
        elif self.mode == "per_node":
            self.raw_step = None
            self.raw_node = make_param((self.N,), raw_init)
        elif self.mode == "per_node_horizon":
            # factorized (cheap): init step/node to 0 so initial wg ~= sigmoid(raw0)
            self.raw_step = make_param((self.O,), 0.0)
            self.raw_node = make_param((self.N,), 0.0)
        else:
            raise ValueError(f"Unknown fusion_mode: {mode} (expected global|per_horizon|per_node|per_node_horizon)")

    @staticmethod
    def _softplus_ratio(raw: torch.Tensor) -> torch.Tensor:
        """Map raw logits to (0,1): w = softplus(raw) / (1 + softplus(raw))."""
        u = torch.nn.functional.softplus(raw)
        return u / (1.0 + u)

    def forward(self) -> Dict[str, torch.Tensor]:
        raw0 = self.raw0 if torch.is_tensor(self.raw0) else torch.as_tensor(self.raw0)
        if self.mode == "global":
            raw = raw0
        elif self.mode == "per_horizon":
            raw = self.raw_step if torch.is_tensor(self.raw_step) else torch.as_tensor(self.raw_step)
        elif self.mode == "per_node":
            raw = self.raw_node if torch.is_tensor(self.raw_node) else torch.as_tensor(self.raw_node)
        else:  # per_node_horizon
            rs = self.raw_step if torch.is_tensor(self.raw_step) else torch.as_tensor(self.raw_step)  # [O]
            rn = self.raw_node if torch.is_tensor(self.raw_node) else torch.as_tensor(self.raw_node)  # [N]
            raw = raw0 + rs.unsqueeze(-1) + rn.unsqueeze(0)  # [O,N]

        w_graph = self._softplus_ratio(raw)
        if not (self.w_min == 0.0 and self.w_max == 1.0):
            w_graph = self.w_min + (self.w_max - self.w_min) * w_graph

        w_base = 1.0 - w_graph
        return {"w_base": w_base, "w_graph": w_graph}


def _expand_weight(w: torch.Tensor, B: int, O: int, N: int, D: int) -> torch.Tensor:
    """
    Expand a weight tensor to [B,O,N,1] (broadcastable to [B,O,N,D]).
    Accepted shapes:
      scalar: []
      per_horizon: [O]
      per_node: [N]
      per_node_horizon: [O,N]
    """
    if w.ndim == 0:
        return w.view(1, 1, 1, 1).expand(B, O, N, 1)
    if w.ndim == 1 and w.shape[0] == O:
        return w.view(1, O, 1, 1).expand(B, O, N, 1)
    if w.ndim == 1 and w.shape[0] == N:
        return w.view(1, 1, N, 1).expand(B, O, N, 1)
    if w.ndim == 2 and w.shape == (O, N):
        return w.view(1, O, N, 1).expand(B, O, N, 1)
    raise ValueError(f"Unsupported fusion weight shape: {tuple(w.shape)}")


def _augment_decoder_features(
    F: torch.Tensor,                      # [B,O,N,D]
    node_emb: Optional[torch.Tensor],     # [N,E]
    step_emb: Optional[torch.Tensor],     # [O,S]
    use_node: bool,
    use_step: bool,
) -> torch.Tensor:
    B, O, N, D = F.shape
    parts = [F]
    if use_node and (node_emb is not None):
        ne = node_emb.view(1, 1, N, -1).expand(B, O, N, -1).to(F.dtype)
        parts.append(ne)
    if use_step and (step_emb is not None):
        se = step_emb.view(1, O, 1, -1).expand(B, O, N, -1).to(F.dtype)
        parts.append(se)
    if len(parts) == 1:
        return F
    return torch.cat(parts, dim=-1)


# =========================================================
# MyModel
# =========================================================

class MyModel(nn.Module):
    """
    MyModel (v5): Backbone-H + interpretable Spatial/Time modules + convex fusion + internal loss.

    Key upgrades over the earlier version (to break the 21.x plateau you observed):
      1) Node embedding (captures node heterogeneity; critical on PEMS07 with N=883).
      2) Step/horizon embedding (captures lead-time heterogeneity; critical for multi-step O=12).
      3) Fusion granularity options (global / per_horizon / per_node / factorized per_node_horizon).
         This addresses the "global wg is a compromise" issue you identified by Step A plateau.

    Runner compatibility:
      - forward_return includes "loss" when targets are provided and compute_loss_in_forward=True.
    """

    def __init__(self, cfg: MyModelConfig):
        super().__init__()
        self.cfg = cfg

        self.L = int(cfg.input_len)
        self.O = int(cfg.output_len)
        self.N = int(cfg.num_features)
        self.T = int(cfg.num_timestamps)

        self.last_value_centering = bool(cfg.last_value_centering)
        self.dropout = nn.Dropout(float(cfg.dropout))

        # ---- identity embeddings ----
        self.node_emb_dim = int(getattr(cfg, "node_emb_dim", 0))
        self.step_emb_dim = int(getattr(cfg, "step_emb_dim", 0))
        self.node_emb_in_backbone = bool(getattr(cfg, "node_emb_in_backbone", True))
        self.node_emb_in_decoder = bool(getattr(cfg, "node_emb_in_decoder", True))
        self.step_emb_in_decoder = bool(getattr(cfg, "step_emb_in_decoder", True))
        self.step_emb_in_time = bool(getattr(cfg, "step_emb_in_time", True))
        self.step_emb_in_graph = bool(getattr(cfg, "step_emb_in_graph", True))

        self.node_emb = None
        self.node_emb_drop = nn.Dropout(float(getattr(cfg, "node_emb_dropout", 0.0)))
        if self.node_emb_dim > 0:
            self.node_emb = nn.Embedding(self.N, self.node_emb_dim)
            nn.init.normal_(self.node_emb.weight, mean=0.0, std=0.02)

        self.step_emb = None
        if self.step_emb_dim > 0:
            self.step_emb = nn.Embedding(self.O, self.step_emb_dim)
            nn.init.normal_(self.step_emb.weight, mean=0.0, std=0.02)

        self.node_bias = None
        if bool(getattr(cfg, "node_bias", False)):
            self.node_bias = nn.Parameter(torch.zeros(self.N))

        # ---- trunk ----
        self.backbone = NodeTemporalBackbone(cfg)
        D = int(self.backbone.out_dim)

        # ---- spatial module ----
        self.graph = None
        if bool(cfg.enable_dynamic_graph):
            self.graph = LowRankDynamicKernelOnFeatures(
                num_features=self.N,
                feat_dim=D,
                rank=int(cfg.graph_rank),
                num_timestamps=self.T,
                use_output_timestamps=bool(cfg.graph_use_output_timestamps) and bool(cfg.use_output_timestamps),
                hidden_size=int(cfg.graph_scale_hidden_size),
                dropout=float(cfg.graph_scale_dropout),
                nonnegative_basis=bool(cfg.graph_nonnegative_basis),
                normalize=bool(cfg.graph_normalize),
                alpha=float(cfg.graph_alpha),
                step_emb_dim=self.step_emb_dim,
                use_step_embedding=self.step_emb_in_graph and self.step_emb_dim > 0,
            )

        # ---- fusion ----
        self.fusion = ConvexGraphFusion(
            learnable=bool(cfg.fusion_learnable),
            raw_init=float(cfg.fusion_raw_init),
            mode=str(getattr(cfg, "fusion_mode", "global")),
            num_nodes=self.N,
            num_horizons=self.O,
            w_min=float(getattr(cfg, "fusion_w_min", 0.0)),
            w_max=float(getattr(cfg, "fusion_w_max", 1.0)),
        )

        # ---- time module ----
        self.time_effect = None
        if bool(cfg.enable_time_effect) and bool(cfg.use_output_timestamps):
            self.time_effect = FourierTimeEffect(
                feat_dim=D,
                node_emb_dim=self.node_emb_dim if self.node_emb_dim > 0 else 0,
                step_emb_dim=self.step_emb_dim if (self.step_emb_in_time and self.step_emb_dim > 0) else 0,
                K_tod=int(cfg.time_tod_harmonics),
                K_dow=int(cfg.time_dow_harmonics),
                coef_hidden=int(cfg.time_coef_hidden),
                dropout=float(cfg.time_coef_dropout),
                use_step_film=self.step_emb_in_time and self.step_emb_dim > 0,
            )

        # ---- decoder heads ----
        dec_dim = D
        if self.node_emb_dim > 0 and self.node_emb_in_decoder:
            dec_dim += self.node_emb_dim
        if self.step_emb_dim > 0 and self.step_emb_in_decoder:
            dec_dim += self.step_emb_dim

        self.mu_head = nn.Linear(dec_dim, 1, bias=True)

        # probabilistic head
        self.likelihood = str(cfg.likelihood).lower()
        self.min_scale = float(cfg.min_scale)
        self.scale_head = None
        self.studentt_df_param = None
        self.studentt_df_min = float(cfg.studentt_df_min)

        if self.likelihood in {"gaussian", "studentt", "laplace", "lognormal", "gamma", "negbinom"}:
            self.scale_head = nn.Linear(dec_dim, 1, bias=True)

            if self.likelihood == "studentt":
                mode = str(cfg.studentt_df_mode).lower()
                if mode == "fixed":
                    self.register_buffer("_studentt_df_fixed", torch.tensor(float(cfg.studentt_df_init)))
                elif mode == "learned_global":
                    init = float(cfg.studentt_df_init)
                    x0 = math.log(math.expm1(max(init - self.studentt_df_min, 1e-3)))
                    self.studentt_df_param = nn.Parameter(torch.tensor(x0))
                else:
                    raise ValueError(f"studentt_df_mode must be fixed|learned_global, got {mode}")

        # quantile head
        self.quantiles = None
        self.q_levels = None
        self.quantile_monotone = bool(cfg.quantile_monotone)
        self.quantile_pred_level = float(cfg.quantile_pred_level)
        self.quantile_crossing_penalty = float(cfg.quantile_crossing_penalty)

        self.q0_head = None
        self.qdelta_head = None
        self.qall_head = None

        if self.likelihood == "quantile":
            q = sorted(set(float(x) for x in cfg.quantiles))
            if len(q) < 2:
                raise ValueError("quantiles must have at least 2 values.")
            for x in q:
                if not (0.0 < x < 1.0):
                    raise ValueError(f"Invalid quantile {x}, should be in (0,1).")
            self.quantiles = q
            self.register_buffer("q_levels", torch.tensor(q), persistent=False)
            Q = len(q)

            if self.quantile_monotone:
                self.q0_head = nn.Linear(dec_dim, 1, bias=True)
                self.qdelta_head = nn.Linear(dec_dim, Q - 1, bias=True)
            else:
                self.qall_head = nn.Linear(dec_dim, Q, bias=True)

        if self.likelihood not in {
            "none", "gaussian", "studentt", "laplace", "quantile", "lognormal", "gamma", "negbinom"
        }:
            raise ValueError(f"Unknown likelihood: {self.likelihood}")

        # ---- training loss config (internal) ----
        self.compute_loss_in_forward = bool(cfg.compute_loss_in_forward)
        self.point_loss = str(cfg.point_loss).lower()
        self.huber_delta = float(cfg.huber_delta)
        self.lambda_point = float(cfg.lambda_point)
        self.lambda_nll = float(cfg.lambda_nll)
        self.loss_eps = float(cfg.loss_eps)
        self.loss_check_domain = bool(cfg.loss_check_domain)

        self.reg_weights = RegWeights(
            reg_graph_orth=float(cfg.reg_graph_orth),
            reg_graph_l1=float(cfg.reg_graph_l1),
            reg_graph_scale_smooth=float(cfg.reg_graph_scale_smooth),
            reg_fusion_l1=float(cfg.reg_fusion_l1),
        )

        # output controls
        self.return_distribution = bool(cfg.return_distribution)
        self.return_interpretation = bool(cfg.return_interpretation)
        self.return_components = bool(cfg.return_components)

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
        if inputs.ndim != 3:
            raise ValueError(f"Expected inputs [B,L,N], got {tuple(inputs.shape)}")
        B, L, N = inputs.shape
        if L != self.L or N != self.N:
            raise ValueError(f"Input shape mismatch: got L={L},N={N}, expected L={self.L},N={self.N}")

        # node/step embeddings on correct device
        node_emb = None
        if self.node_emb is not None:
            node_ids = torch.arange(self.N, device=inputs.device)
            node_emb = self.node_emb_drop(self.node_emb(node_ids))  # [N,E]

        step_emb = None
        if self.step_emb is not None:
            step_ids = torch.arange(self.O, device=inputs.device)
            step_emb = self.step_emb(step_ids)  # [O,S]

        # (1) last-value centering
        if self.last_value_centering:
            last = inputs[:, -1, :]                 # [B,N]
            x0 = inputs - last.unsqueeze(1)         # [B,L,N]
        else:
            last = None
            x0 = inputs

        x0 = self.dropout(x0)

        # (2) trunk backbone -> H [B,N,D]
        ts_in = inputs_timestamps if bool(self.cfg.use_input_timestamps) else None
        H = self.backbone(x0, ts_in, node_emb=node_emb if self.node_emb_in_backbone else None)

        O = self.O
        H_base = H.unsqueeze(1).expand(-1, O, -1, -1)  # [B,O,N,D]

        # (3) spatial module
        graph_info: Dict[str, torch.Tensor] = {}
        if self.graph is not None:
            H_graph, graph_info = self.graph(H, ts_out=targets_timestamps, O=O, step_emb=step_emb if self.step_emb_in_graph else None)
        else:
            H_graph = H_base

        # (4) fusion weights and fused features
        if self.graph is not None:
            fw = self.fusion()
            w_base_raw, w_graph_raw = fw["w_base"], fw["w_graph"]
        else:
            w_base_raw = H.new_tensor(1.0)
            w_graph_raw = H.new_tensor(0.0)
            fw = {"w_base": w_base_raw, "w_graph": w_graph_raw}

        w_base = _expand_weight(w_base_raw, B, O, N, H.shape[-1])   # [B,O,N,1]
        w_graph = _expand_weight(w_graph_raw, B, O, N, H.shape[-1]) # [B,O,N,1]

        F = w_base * H_base + w_graph * H_graph  # [B,O,N,D]

        # (5) decoder input augmentation
        F_dec = _augment_decoder_features(
            F,
            node_emb=node_emb,
            step_emb=step_emb,
            use_node=(self.node_emb is not None and self.node_emb_in_decoder),
            use_step=(self.step_emb is not None and self.step_emb_in_decoder),
        )

        mu_feat = self.mu_head(F_dec).squeeze(-1)  # [B,O,N]

        # optional decomposition (compute only if user asks)
        mu_base = None
        mu_graph = None
        if self.return_components or self.return_interpretation:
            Hb_dec = _augment_decoder_features(
                H_base,
                node_emb=node_emb,
                step_emb=step_emb,
                use_node=(self.node_emb is not None and self.node_emb_in_decoder),
                use_step=(self.step_emb is not None and self.step_emb_in_decoder),
            )
            Hg_dec = _augment_decoder_features(
                H_graph,
                node_emb=node_emb,
                step_emb=step_emb,
                use_node=(self.node_emb is not None and self.node_emb_in_decoder),
                use_step=(self.step_emb is not None and self.step_emb_in_decoder),
            )
            mu_base = self.mu_head(Hb_dec).squeeze(-1)
            mu_graph = self.mu_head(Hg_dec).squeeze(-1)

        # (6) additive time effect
        time_info: Dict[str, torch.Tensor] = {}
        if self.time_effect is not None:
            if targets_timestamps is None:
                raise ValueError("enable_time_effect=True but targets_timestamps is None.")
            time_total, time_info = self.time_effect(
                H,
                targets_timestamps,
                node_emb=node_emb if self.node_emb_dim > 0 else None,
                step_emb=step_emb if (self.step_emb_in_time and self.step_emb_dim > 0) else None,
            )
            mu_delta = mu_feat + time_total
        else:
            mu_delta = mu_feat

        mu = mu_delta

        # node bias (optional)
        if self.node_bias is not None:
            mu = mu + self.node_bias.view(1, 1, -1).to(mu.dtype)

        # restore last value if using delta-style
        if last is not None:
            mu = mu + last.unsqueeze(1).expand(-1, O, -1)

        # (7) distribution parameters (decoded from F_dec)
        dist = self.likelihood
        dist_params: Dict[str, torch.Tensor] = {}
        prediction = mu

        if dist in {"gaussian", "studentt", "laplace", "lognormal", "gamma", "negbinom"}:
            log_s = self.scale_head(F_dec).squeeze(-1)  # [B,O,N]
            scale = torch.nn.functional.softplus(log_s) + self.min_scale

            if dist in {"gaussian", "studentt", "laplace"}:
                dist_params = {"mu": mu, "scale": scale}
                if dist == "studentt":
                    if hasattr(self, "_studentt_df_fixed"):
                        df = self._studentt_df_fixed.to(mu.device).to(mu.dtype)
                    else:
                        df = self.studentt_df_min + torch.nn.functional.softplus(self.studentt_df_param)
                        df = df.to(mu.device).to(mu.dtype)
                    dist_params["df"] = df
                prediction = mu

            elif dist == "lognormal":
                dist_params = {"mu": mu, "scale": scale}
                prediction = torch.exp(mu + 0.5 * scale ** 2)

            elif dist == "gamma":
                mu_pos = torch.nn.functional.softplus(mu) + 1e-6
                shape = torch.nn.functional.softplus(scale) + 1e-6
                dist_params = {"mu_pos": mu_pos, "shape": shape}
                prediction = mu_pos

            else:  # negbinom
                mu_pos = torch.nn.functional.softplus(mu) + 1e-6
                total_count = torch.nn.functional.softplus(scale) + 1e-6
                dist_params = {"mu_pos": mu_pos, "total_count": total_count}
                prediction = mu_pos

        elif dist == "quantile":
            Q = int(self.q_levels.numel())
            if self.quantile_monotone:
                q0 = self.q0_head(F_dec).squeeze(-1)
                d = self.qdelta_head(F_dec)
                d = torch.nn.functional.softplus(d)
                qs = [q0.unsqueeze(-1)]
                q_prev = q0
                for i in range(Q - 1):
                    q_prev = q_prev + d[..., i]
                    qs.append(q_prev.unsqueeze(-1))
                q_all = torch.cat(qs, dim=-1)
            else:
                q_all = self.qall_head(F_dec)

            dist_params = {"quantiles": q_all, "q_levels": self.q_levels.to(device=mu.device, dtype=mu.dtype)}

            idx = int(torch.argmin(torch.abs(self.q_levels.to(mu.device) - self.quantile_pred_level)).item())
            prediction = q_all[..., idx]

            if (not self.quantile_monotone) and self.quantile_crossing_penalty > 0:
                diffs = q_all[..., 1:] - q_all[..., :-1]
                dist_params["cross_penalty"] = self.quantile_crossing_penalty * torch.relu(-diffs).mean()

        elif dist == "none":
            prediction = mu
        else:
            raise ValueError(f"Unknown likelihood: {dist}")

        # (8) output dict
        out: Dict = {"prediction": prediction}

        if self.return_distribution and dist != "none":
            out["dist_name"] = dist
            out["dist_params"] = dist_params

        if self.return_interpretation:
            # store raw fusion weights (not expanded)
            out["fusion_weights"] = {"w_base": w_base_raw.detach(), "w_graph": w_graph_raw.detach()}
            if self.graph is not None:
                out.update(graph_info)
            if self.time_effect is not None:
                out.update(time_info)

        if self.return_components:
            comps: Dict[str, torch.Tensor] = {
                "w_base": w_base_raw.detach(),
                "w_graph": w_graph_raw.detach(),
                "mu": mu,
            }
            if mu_base is not None:
                comps["mu_base"] = mu_base + (last.unsqueeze(1) if last is not None else 0.0)
            if mu_graph is not None:
                comps["mu_graph"] = mu_graph + (last.unsqueeze(1) if last is not None else 0.0)
            comps["mu_feat"] = mu_feat + (last.unsqueeze(1) if last is not None else 0.0)
            if self.time_effect is not None:
                comps["time_tod"] = time_info.get("time_tod")
                comps["time_dow"] = time_info.get("time_dow")
                comps["time_total"] = time_info.get("time_total")
            if last is not None:
                comps["last_value"] = last
            if self.node_bias is not None:
                comps["node_bias"] = self.node_bias.detach()
            out["components"] = comps

        # (9) loss inside forward (runner compatibility)
        if self.compute_loss_in_forward and (targets is not None) and (targets.numel() > 0):
            loss_inputs: Dict = {"prediction": prediction}
            if dist != "none":
                loss_inputs["dist_name"] = dist
                loss_inputs["dist_params"] = dist_params

            # regularization signals
            loss_inputs["fusion_weights"] = {"w_base": w_base_raw, "w_graph": w_graph_raw}
            if self.graph is not None:
                loss_inputs["graph_basis"] = graph_info["graph_basis"]
                loss_inputs["graph_scales"] = graph_info["graph_scales"]

            loss_dict = compute_total_loss(
                outputs=loss_inputs,
                targets=targets,
                targets_mask=targets_mask,
                point_loss=self.point_loss,
                huber_delta=self.huber_delta,
                lambda_point=self.lambda_point,
                lambda_nll=self.lambda_nll,
                reg_weights=self.reg_weights,
                eps=self.loss_eps,
                check_domain=self.loss_check_domain,
            )
            out.update(loss_dict)

        return out
