
from dataclasses import dataclass, field
from typing import Sequence

from basicts.configs import BasicTSModelConfig


@dataclass
class MyModelConfig(BasicTSModelConfig):
    """
    MyModel config (Backbone-H + Spatial/Time/Distribution modules + Convex Fusion).

    Fixed shapes:
      inputs:             [B, L, N]
      targets:            [B, O, N]
      inputs_timestamps:  [B, L, T]
      targets_timestamps: [B, O, T]

    Timestamp convention in this project:
      - timestamps are already normalized to [0, 1] BEFORE entering the model.
      - timestamp_sizes (e.g. [288, 7]) is meta-info for the original discrete sizes
        and is NOT used for embeddings in this implementation.
    """

    # ---- required ----
    input_len: int = field(default=None)
    output_len: int = field(default=None)
    num_features: int = field(default=None)     # number of nodes N
    num_timestamps: int = field(default=2)      # e.g. T=2: [tod_norm, dow_norm]

    # meta only (not used for embeddings when timestamps are normalized)
    timestamp_sizes: Sequence[int] = field(default=(288, 7))

    # ---- preprocessing ----
    last_value_centering: bool = field(default=True)
    dropout: float = field(default=0.0)

    # ---- trunk backbone ----
    backbone_type: str = field(default="gru", metadata={"help": "lstm|gru|transformer"})
    backbone_layers: int = field(default=2)
    backbone_hidden_size: int = field(default=64)
    backbone_dropout: float = field(default=0.0)

    # transformer-only knobs
    transformer_nhead: int = field(default=4)
    transformer_ffn_ratio: float = field(default=4.0)
    transformer_norm_first: bool = field(default=True)
    transformer_use_positional_encoding: bool = field(default=True)

    # ---- timestamps usage ----
    use_input_timestamps: bool = field(default=False)   # concat to backbone input
    use_output_timestamps: bool = field(default=True)   # used by spatial/time modules

    # =========================
    # Module A) Spatial interpretability (adjacency-free, avoids N×N):
    #   A(b,o) = B diag(s(b,o)) B^T   (PSD, rank=r)
    # =========================
    enable_dynamic_graph: bool = field(default=True)
    graph_rank: int = field(default=16)
    graph_alpha: float = field(default=0.5)             # convex update coefficient in [0,1]
    graph_normalize: bool = field(default=True)
    graph_nonnegative_basis: bool = field(default=True)
    graph_use_output_timestamps: bool = field(default=True)
    graph_scale_hidden_size: int = field(default=64)
    graph_scale_dropout: float = field(default=0.0)

    # statistical regularization (interpretability / identifiability)
    reg_graph_orth: float = field(default=0.0)          # B^T B ~ I
    reg_graph_l1: float = field(default=0.0)            # sparsity of B
    reg_graph_scale_smooth: float = field(default=0.0)  # smooth s across horizons

    # =========================
    # Fusion (Method A) between {base, graph}:
    #   F = w0*H_base + wg*H_graph, w0,wg>0, w0+wg=1.
    # =========================
    fusion_learnable: bool = field(default=True)
    fusion_raw_init: float = field(default=0.0)         # raw init for graph weight before softplus
    reg_fusion_l1: float = field(default=0.0)

    # =========================
    # Module B) Time interpretability (stronger, Fourier basis):
    #   b_time(b,o,n) = <c_tod(H_{b,n}), phi_tod(tod_{b,o})> + <c_dow(H_{b,n}), phi_dow(dow_{b,o})>
    # =========================
    enable_time_effect: bool = field(default=True)

    # number of Fourier harmonics (k=1..K)
    time_tod_harmonics: int = field(default=4)
    time_dow_harmonics: int = field(default=2)

    # coefficients network on H:
    #   if 0 -> linear map D->P; if >0 -> 2-layer MLP with this hidden size.
    time_coef_hidden: int = field(default=0)
    time_coef_dropout: float = field(default=0.0)

    # =========================
    # Module C) Distribution interpretability (learn distribution parameters)
    # =========================
    likelihood: str = field(
        default="gaussian",
        metadata={"help": "none|gaussian|studentt|laplace|quantile|lognormal|gamma|negbinom"},
    )

    # scale head params
    min_scale: float = field(default=1e-3)

    # student-t df
    studentt_df_mode: str = field(default="learned_global", metadata={"help": "fixed|learned_global"})
    studentt_df_init: float = field(default=10.0)
    studentt_df_min: float = field(default=2.1)

    # quantile
    quantiles: Sequence[float] = field(default=(0.1, 0.5, 0.9))
    quantile_monotone: bool = field(default=True)
    quantile_pred_level: float = field(default=0.5)
    quantile_crossing_penalty: float = field(default=0.0)

    # =========================
    # Training loss (computed inside model.forward for runner compatibility)
    # =========================
    point_loss: str = field(default="mae")              # mae|mse|huber
    huber_delta: float = field(default=1.0)
    lambda_point: float = field(default=1.0)
    lambda_nll: float = field(default=1.0)

    loss_eps: float = field(default=1e-6)
    loss_check_domain: bool = field(default=True)

    compute_loss_in_forward: bool = field(default=True)  # to match runner: forward_return["loss"]

    # ---- outputs ----
    return_distribution: bool = field(default=True)
    return_interpretation: bool = field(default=True)
    return_components: bool = field(default=True)
