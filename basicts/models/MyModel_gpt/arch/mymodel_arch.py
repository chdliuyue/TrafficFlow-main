
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
        in_dim = 1 + (self.T if self.use_in_ts else 0)

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

    def forward(self, values: torch.Tensor, ts_in: Optional[torch.Tensor]) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(f"Expected values [B,L,N], got {tuple(values.shape)}")
        B, L, N = values.shape
        if L != self.L or N != self.N:
            raise ValueError(f"Input shape mismatch: got L={L},N={N}, expected L={self.L},N={self.N}")

        # per-node sequence: [B,N,L,1]
        v = values.permute(0, 2, 1).unsqueeze(-1)

        if self.use_in_ts:
            if ts_in is None:
                raise ValueError("use_input_timestamps=True but inputs_timestamps is None.")
            if ts_in.shape[:2] != (B, L):
                raise ValueError(f"inputs_timestamps should be [B,L,T], got {tuple(ts_in.shape)}")
            ts = ts_in.unsqueeze(1).expand(-1, N, -1, -1).float()
            x = torch.cat([v, ts], dim=-1)  # [B,N,L,1+T]
        else:
            x = v

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
# Module B) Time interpretability: Fourier basis (stronger & decomposable)
# =========================================================

class FourierTimeEffect(nn.Module):
    r"""
    Time effect with a fixed Fourier basis and H-conditioned coefficients:

      b_time(b,o,n) = <c_tod(H_{b,n}), phi_tod(tod_{b,o})> + <c_dow(H_{b,n}), phi_dow(dow_{b,o})>

    where:
      phi_tod(t) = [sin(2πkt), cos(2πkt)]_{k=1..K_tod}   (mean-zero, periodic with period 1)
      phi_dow(d) = [sin(2πkd), cos(2πkd)]_{k=1..K_dow}

    - tod, dow are assumed normalized to [0,1] in the pipeline.
    - Interpretation:
        * c_tod, c_dow are amplitude/phase-like coefficients (conditioned on current traffic state H).
        * The time effect is an additive component in the mean, so it is directly attributable.
    """

    def __init__(self, feat_dim: int, K_tod: int, K_dow: int, coef_hidden: int = 0, dropout: float = 0.0):
        super().__init__()
        self.D = int(feat_dim)
        self.K_tod = int(K_tod)
        self.K_dow = int(K_dow)

        P_tod = 2 * self.K_tod
        P_dow = 2 * self.K_dow

        def make_coef_net(P: int):
            if P == 0:
                return None
            if int(coef_hidden) <= 0:
                return nn.Linear(self.D, P, bias=True)
            return nn.Sequential(
                nn.Linear(self.D, int(coef_hidden)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(coef_hidden), P),
            )

        self.coef_tod = make_coef_net(P_tod)
        self.coef_dow = make_coef_net(P_dow)

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

    def forward(self, H: torch.Tensor, ts_out: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        H:     [B,N,D]
        ts_out:[B,O,T] where ts_out[...,0]=tod_norm, ts_out[...,1]=dow_norm (both in [0,1])
        returns:
          time_total: [B,O,N]
          info: dict with decomposition
        """
        if ts_out is None:
            raise ValueError("targets_timestamps is required for time effect.")

        B, N, D = H.shape
        Bt, O, T = ts_out.shape
        if Bt != B:
            raise ValueError("Batch mismatch between H and targets_timestamps")
        if T < 2:
            raise ValueError("FourierTimeEffect expects at least 2 timestamp dims: [tod_norm, dow_norm].")

        tod = ts_out[..., 0].float()  # [B,O]
        dow = ts_out[..., 1].float()  # [B,O]

        basis_tod = self._fourier_basis(tod, self.K_tod)  # [B,O,2K_tod]
        basis_dow = self._fourier_basis(dow, self.K_dow)  # [B,O,2K_dow]

        # coefficients from H
        coef_tod = None if self.coef_tod is None else self.coef_tod(H)  # [B,N,2K_tod]
        coef_dow = None if self.coef_dow is None else self.coef_dow(H)  # [B,N,2K_dow]

        time_tod = H.new_zeros(B, O, N)
        time_dow = H.new_zeros(B, O, N)

        if coef_tod is not None and basis_tod.numel() > 0:
            time_tod = torch.einsum("bop,bnp->bon", basis_tod, coef_tod)
        if coef_dow is not None and basis_dow.numel() > 0:
            time_dow = torch.einsum("bop,bnp->bon", basis_dow, coef_dow)

        time_total = time_tod + time_dow

        info = {
            "time_tod": time_tod,               # [B,O,N]
            "time_dow": time_dow,               # [B,O,N]
            "time_total": time_total,           # [B,O,N]
            "time_coef_tod": coef_tod,          # [B,N,2K_tod] or None
            "time_coef_dow": coef_dow,          # [B,N,2K_dow] or None
            "time_basis_tod": basis_tod,        # [B,O,2K_tod]
            "time_basis_dow": basis_dow,        # [B,O,2K_dow]
        }
        return time_total, info


# =========================================================
# Module A) Spatial interpretability: low-rank PSD dynamic kernel on H (no N×N)
# =========================================================

class LowRankDynamicKernelOnFeatures(nn.Module):
    r"""
    Adjacency-free dynamic interaction on features H via a low-rank PSD operator.

    For each horizon o:
        A(b,o) = B diag(s(b,o)) B^T,   s(b,o) >= 0  =>  A(b,o) PSD, rank<=r.

    We DO NOT materialize A (N×N). Compute:
        M = A H = B ( s ⊙ (B^T H) )     with O(N*r*D).

    Optional degree-like normalization keeps scale stable.

    Output (module-internal convex update):
        H_graph = H_base + alpha*(M - H_base), alpha in [0,1].
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
        eps: float = 1e-6,
    ):
        super().__init__()
        self.N = int(num_features)
        self.D = int(feat_dim)
        self.r = int(rank)
        self.T = int(num_timestamps)
        self.use_out_ts = bool(use_output_timestamps)
        self.nonneg = bool(nonnegative_basis)
        self.normalize = bool(normalize)
        self.eps = float(eps)

        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

        self._basis = nn.Parameter(torch.empty(self.N, self.r))
        nn.init.xavier_uniform_(self._basis)

        in_dim = self.r + (self.T if self.use_out_ts else 0)
        self.scale_mlp = nn.Sequential(
            nn.Linear(in_dim, int(hidden_size)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), self.r),
        )

    def basis(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._basis) if self.nonneg else self._basis

    def forward(self, H: torch.Tensor, ts_out: Optional[torch.Tensor], O: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if H.ndim != 3:
            raise ValueError(f"Expected H [B,N,D], got {tuple(H.shape)}")
        Bsz, N, D = H.shape
        if N != self.N or D != self.D:
            raise ValueError(f"H shape mismatch: got {tuple(H.shape)}, expected [B,{self.N},{self.D}]")

        Bmat = self.basis()                                   # [N,r]
        U = torch.einsum("bnd,nr->brd", H, Bmat)              # [B,r,D]
        ctx = torch.sqrt(torch.mean(U ** 2, dim=-1) + self.eps)  # [B,r]

        if self.use_out_ts:
            if ts_out is None:
                raise ValueError("graph_use_output_timestamps=True but targets_timestamps is None.")
            if ts_out.shape[0] != Bsz or ts_out.shape[1] != O:
                raise ValueError(f"targets_timestamps should be [B,O,T], got {tuple(ts_out.shape)}")
            feat = torch.cat([ts_out.float(), ctx.unsqueeze(1).expand(-1, O, -1)], dim=-1)  # [B,O,T+r]
        else:
            feat = ctx.unsqueeze(1).expand(-1, O, -1)  # [B,O,r]

        s = torch.nn.functional.softplus(self.scale_mlp(feat)) + self.eps  # [B,O,r] >=0

        SU = s.unsqueeze(-1) * U.unsqueeze(1)              # [B,O,r,D]
        M = torch.einsum("bord,nr->bond", SU, Bmat)        # [B,O,N,D]

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
# Fusion: Method A convex weights (base + graph)
# =========================================================

class ConvexGraphFusion(nn.Module):
    """
    Strict convex weights over {base, graph}:

      u = softplus(raw) >= 0
      w_graph = u / (1 + u)
      w_base  = 1 / (1 + u)

    => w_base, w_graph > 0 and w_base + w_graph = 1.
    """

    def __init__(self, learnable: bool = True, raw_init: float = 0.0):
        super().__init__()
        init = torch.tensor(float(raw_init), dtype=torch.float)
        if learnable:
            self.raw = nn.Parameter(init)
        else:
            self.register_buffer("raw", init, persistent=True)

    def forward(self) -> Dict[str, torch.Tensor]:
        u = torch.nn.functional.softplus(self.raw)
        w_graph = u / (1.0 + u)
        w_base = 1.0 - w_graph
        return {"w_base": w_base, "w_graph": w_graph}


# =========================================================
# Main model
# =========================================================

class MyModel(nn.Module):
    """
    MyModel: Backbone-H + three interpretable modules + convex fusion + internal loss.

    Pipeline (high level):
      1) (optional) last-value centering: X -> X - X_last
      2) trunk backbone: H = f_theta(X, S_in)            [B,N,D]
      3) spatial module: H_graph = G(H, S_out)           [B,O,N,D]   (avoids N×N)
      4) convex fusion:  F = w0*H_base + wg*H_graph      [B,O,N,D]
      5) mean:           mu = Dec_mu(F) + time_effect + restore_last
      6) distribution:   decode parameters from F
      7) if targets provided and compute_loss_in_forward=True:
            out["loss"] = total loss (point + NLL + regularization)
         so runner can directly read forward_return["loss"].
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

        # trunk
        self.backbone = NodeTemporalBackbone(cfg)
        D = int(self.backbone.out_dim)

        # spatial module
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
            )

        # fusion weights
        self.fusion = ConvexGraphFusion(
            learnable=bool(cfg.fusion_learnable),
            raw_init=float(cfg.fusion_raw_init),
        )

        # time module (strong interpretability)
        self.time_effect = None
        if bool(cfg.enable_time_effect) and bool(cfg.use_output_timestamps):
            self.time_effect = FourierTimeEffect(
                feat_dim=D,
                K_tod=int(cfg.time_tod_harmonics),
                K_dow=int(cfg.time_dow_harmonics),
                coef_hidden=int(cfg.time_coef_hidden),
                dropout=float(cfg.time_coef_dropout),
            )

        # decoding heads (linear heads keep decomposition clear)
        self.mu_head = nn.Linear(D, 1, bias=True)

        # probabilistic head
        self.likelihood = str(cfg.likelihood).lower()
        self.min_scale = float(cfg.min_scale)
        self.scale_head = None
        self.studentt_df_param = None
        self.studentt_df_min = float(cfg.studentt_df_min)

        if self.likelihood in {"gaussian", "studentt", "laplace", "lognormal", "gamma", "negbinom"}:
            self.scale_head = nn.Linear(D, 1, bias=True)

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
                self.q0_head = nn.Linear(D, 1, bias=True)
                self.qdelta_head = nn.Linear(D, Q - 1, bias=True)
            else:
                self.qall_head = nn.Linear(D, Q, bias=True)

        if self.likelihood not in {
            "none", "gaussian", "studentt", "laplace", "quantile", "lognormal", "gamma", "negbinom"
        }:
            raise ValueError(f"Unknown likelihood: {self.likelihood}")

        # training loss config (internal)
        self.compute_loss_in_forward = bool(cfg.compute_loss_in_forward)
        self.point_loss = str(cfg.point_loss).lower()
        self.huber_delta = float(cfg.huber_delta)
        self.lambda_point = float(cfg.lambda_point)
        self.lambda_nll = float(cfg.lambda_nll)
        self.loss_eps = float(cfg.loss_eps)
        self.loss_check_domain = bool(cfg.loss_check_domain)

        # regularization weights passed to myloss
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
        train: bool = False,  # runner may pass this; loss computation depends on targets existence
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> Dict:
        if inputs.ndim != 3:
            raise ValueError(f"Expected inputs [B,L,N], got {tuple(inputs.shape)}")
        B, L, N = inputs.shape
        if L != self.L or N != self.N:
            raise ValueError(f"Input shape mismatch: got L={L},N={N}, expected L={self.L},N={self.N}")

        if bool(self.cfg.use_output_timestamps) and targets_timestamps is None:
            # time/spatial modules may need ts_out; we enforce if configured
            pass

        # (1) last-value centering (delta-style)
        if self.last_value_centering:
            last = inputs[:, -1, :]                 # [B,N]
            x0 = inputs - last.unsqueeze(1)         # [B,L,N]
        else:
            last = None
            x0 = inputs

        x0 = self.dropout(x0)

        # (2) trunk backbone -> H [B,N,D]
        H = self.backbone(x0, inputs_timestamps if self.cfg.use_input_timestamps else None)

        # base horizon feature
        O = self.O
        H_base = H.unsqueeze(1).expand(-1, O, -1, -1)  # [B,O,N,D]

        # (3) spatial module (adjacency-free)
        graph_info: Dict[str, torch.Tensor] = {}
        if self.graph is not None:
            H_graph, graph_info = self.graph(H, ts_out=targets_timestamps, O=O)
        else:
            H_graph = H_base

        # (4) convex fusion (base + graph)
        if self.graph is not None:
            fw = self.fusion()
            w_base, w_graph = fw["w_base"], fw["w_graph"]
        else:
            # no graph => identity
            w_base = H.new_tensor(1.0)
            w_graph = H.new_tensor(0.0)
            fw = {"w_base": w_base, "w_graph": w_graph}

        F = w_base * H_base + w_graph * H_graph  # [B,O,N,D]

        # (5) mean from fused features + additive time effect
        mu_base = self.mu_head(H_base).squeeze(-1)      # [B,O,N]
        mu_graph = self.mu_head(H_graph).squeeze(-1)    # [B,O,N]
        mu_feat = w_base * mu_base + w_graph * mu_graph # [B,O,N] (equal to mu_head(F).squeeze(-1))

        time_info: Dict[str, torch.Tensor] = {}
        if self.time_effect is not None:
            if targets_timestamps is None:
                raise ValueError("enable_time_effect=True but targets_timestamps is None.")
            time_total, time_info = self.time_effect(H, targets_timestamps)
            mu_delta = mu_feat + time_total
        else:
            mu_delta = mu_feat

        mu = mu_delta
        if last is not None:
            mu = mu + last.unsqueeze(1).expand(-1, O, -1)

        # (6) distribution parameters (decoded from F)
        dist = self.likelihood
        dist_params: Dict[str, torch.Tensor] = {}
        prediction = mu

        if dist in {"gaussian", "studentt", "laplace", "lognormal", "gamma", "negbinom"}:
            log_s = self.scale_head(F).squeeze(-1)  # [B,O,N]
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
                prediction = torch.exp(mu + 0.5 * scale ** 2)  # mean in original domain (only if targets > 0)

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
                q0 = self.q0_head(F).squeeze(-1)            # [B,O,N]
                d = self.qdelta_head(F)                     # [B,O,N,Q-1]
                d = torch.nn.functional.softplus(d)
                qs = [q0.unsqueeze(-1)]
                q_prev = q0
                for i in range(Q - 1):
                    q_prev = q_prev + d[..., i]
                    qs.append(q_prev.unsqueeze(-1))
                q_all = torch.cat(qs, dim=-1)               # [B,O,N,Q]
            else:
                q_all = self.qall_head(F)                   # [B,O,N,Q]

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

        # (7) output dict (runner uses out["prediction"] for most metrics; and out["loss"] if asked)
        out: Dict = {"prediction": prediction}

        if self.return_distribution and dist != "none":
            out["dist_name"] = dist
            out["dist_params"] = dist_params

        # optional interpretation payload (kept separate from loss computation)
        if self.return_interpretation:
            out["fusion_weights"] = {"w_base": w_base.detach(), "w_graph": w_graph.detach()}
            if self.graph is not None:
                out.update(graph_info)
            if self.time_effect is not None:
                # include decomposed effects (tod/dow) and coefficients/basis
                out.update(time_info)

        if self.return_components:
            comps: Dict[str, torch.Tensor] = {
                "w_base": w_base.detach(),
                "w_graph": w_graph.detach(),
                "mu_base": mu_base + (last.unsqueeze(1) if last is not None else 0.0),
                "mu_graph": mu_graph + (last.unsqueeze(1) if last is not None else 0.0),
                "mu_feat": mu_feat + (last.unsqueeze(1) if last is not None else 0.0),
                "mu": mu,
            }
            if self.time_effect is not None:
                comps["time_tod"] = time_info["time_tod"]
                comps["time_dow"] = time_info["time_dow"]
                comps["time_total"] = time_info["time_total"]
            if last is not None:
                comps["last_value"] = last
            out["components"] = comps

        # (8) loss computed INSIDE forward for runner compatibility:
        # compute whenever targets exist (train or val), as long as compute_loss_in_forward=True.
        if self.compute_loss_in_forward and (targets is not None) and (targets.numel() > 0):
            # build a compact dict for loss computation (include reg tensors even if not returned)
            loss_inputs: Dict = {"prediction": prediction}

            if dist != "none":
                loss_inputs["dist_name"] = dist
                loss_inputs["dist_params"] = dist_params

            # reg terms
            loss_inputs["fusion_weights"] = {"w_base": w_base, "w_graph": w_graph}
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
            out.update(loss_dict)  # includes out["loss"]

        return out
