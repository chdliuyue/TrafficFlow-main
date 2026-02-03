from dataclasses import dataclass, field
from typing import Sequence

from basicts.configs import BasicTSModelConfig


@dataclass
class MyModelConfig(BasicTSModelConfig):
    """
    MyModel config (Backbone-H + Spatial/Time/Distribution modules + Convex Fusion).

    Fixed shapes in BasicTS:
      inputs:             [B, L, N]
      targets:            [B, O, N]
      inputs_timestamps:  [B, L, T]
      targets_timestamps: [B, O, T]

    Timestamp convention in this project:
      - timestamps are already normalized to [0, 1] BEFORE entering the model.
      - timestamp_sizes (e.g. [288, 7]) is meta-info for the original discrete sizes.

    This config is backward-compatible with previous versions and adds:
      - Node identity embedding (captures node heterogeneity without adjacency).
      - Horizon/step embedding (captures lead-time heterogeneity for multi-step forecasting).
      - Optional decoder conditioning on output timestamps (ts_out) to model interactions μ(H, t) beyond additive time bias.
      - More expressive convex fusion weights (static modes + optional dynamic-per-horizon gating).
      - Optional directed low-rank spatial operator (captures asymmetric influence without building N×N).
    """

    # ---- required ----
    input_len: int = field(default=None)
    output_len: int = field(default=None)
    num_features: int = field(default=None)     # number of nodes N
    num_timestamps: int = field(default=2)      # e.g. T=2: [tod_norm, dow_norm]

    # meta only (timestamps already normalized)
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
    # Identity embeddings (effective on large-N datasets like PEMS07)
    # =========================
    node_emb_dim: int = field(default=0, metadata={"help": "0 disables node embedding; typical: 16/32/64"})
    node_emb_in_backbone: bool = field(default=True, metadata={"help": "concat node embedding to trunk input"})
    node_emb_in_decoder: bool = field(default=True, metadata={"help": "concat node embedding to decoder input"})
    node_emb_dropout: float = field(default=0.0)
    node_bias: bool = field(default=False, metadata={"help": "add a learnable node-wise bias to output mean"})

    step_emb_dim: int = field(default=0, metadata={"help": "0 disables horizon/step embedding; typical: 8/16/32"})
    step_emb_in_decoder: bool = field(default=True, metadata={"help": "concat step embedding to decoder input"})
    step_emb_in_time: bool = field(default=True, metadata={"help": "use step embedding to modulate time coefficients"})
    step_emb_in_graph: bool = field(default=True, metadata={"help": "use step embedding in graph scale network"})

    # =========================
    # NEW: decoder conditioning on output timestamps (ts_out)
    # =========================
    decoder_use_output_timestamps: bool = field(
        default=False,
        metadata={"help": "if True, concat targets_timestamps (normalized) to decoder features; models μ(H,t) interactions."},
    )

    # =========================
    # Module A) Spatial interpretability (adjacency-free, avoids N×N)
    # =========================
    enable_dynamic_graph: bool = field(default=True)

    # operator variant:
    #   - symmetric:  A = B diag(s) B^T   (PSD if s>=0)
    #   - directed:   A = P diag(s) Q^T   (asymmetric, captures direction; still O(N*r*D))
    graph_variant: str = field(default="symmetric", metadata={"help": "symmetric|directed"})

    graph_rank: int = field(default=16)
    graph_alpha: float = field(default=0.5)             # convex update coefficient in [0,1]

    # normalization is meaningful mainly for nonnegative symmetric bases; for signed/directed, keep False.
    graph_normalize: bool = field(default=True)
    graph_nonnegative_basis: bool = field(default=True)

    graph_use_output_timestamps: bool = field(default=True)
    graph_scale_hidden_size: int = field(default=64)
    graph_scale_dropout: float = field(default=0.0)

    # scale activation controls s(b,o):
    #   - softplus: s>=0  (default, PSD-style)
    #   - tanh:     s in [-bound, bound] (signed; useful for directed operator)
    graph_scale_activation: str = field(default="softplus", metadata={"help": "softplus|tanh"})
    graph_scale_bound: float = field(default=1.0, metadata={"help": "bound used when activation=tanh"})

    # statistical regularization (interpretability / identifiability)
    reg_graph_orth: float = field(default=0.0)          # encourage B^T B ~ I (and P^T P, Q^T Q for directed)
    reg_graph_l1: float = field(default=0.0)            # sparsity of basis
    reg_graph_scale_smooth: float = field(default=0.0)  # smooth s across horizons

    # =========================
    # Fusion (Method A) between {base, graph} under convex constraint:
    #   F = w0 * H_base + wg * H_graph, wg in (0,1), w0=1-wg
    #
    # fusion_mode (static):
    #   - global:           wg is a scalar
    #   - per_horizon:      wg(o)
    #   - per_node:         wg(n)
    #   - per_node_horizon: wg(o,n) = sigmoid(raw0 + raw_step[o] + raw_node[n]) (factorized, cheap)
    #
    # fusion_mode (dynamic):
    #   - dynamic_per_horizon: wg(b,o) = g(H, ts_out, step_emb)  (convex, stable)
    # =========================
    fusion_learnable: bool = field(default=True)
    fusion_raw_init: float = field(default=0.0)

    fusion_mode: str = field(
        default="global",
        metadata={"help": "global|per_horizon|per_node|per_node_horizon|dynamic_per_horizon"},
    )
    fusion_w_min: float = field(default=0.0)
    fusion_w_max: float = field(default=1.0)

    # for dynamic_per_horizon:
    fusion_dynamic_ctx_dim: int = field(default=16, metadata={"help": "projected context dim from H for dynamic fusion"})
    fusion_dynamic_hidden: int = field(default=64, metadata={"help": "hidden size of dynamic fusion MLP"})
    fusion_dynamic_dropout: float = field(default=0.0)

    # sparsity penalty on w_graph (implemented in myloss)
    reg_fusion_l1: float = field(default=0.0)

    # =========================
    # Module B) Time interpretability (Fourier basis)
    # =========================
    enable_time_effect: bool = field(default=True)

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

    min_scale: float = field(default=1e-3)

    studentt_df_mode: str = field(default="learned_global", metadata={"help": "fixed|learned_global"})
    studentt_df_init: float = field(default=10.0)
    studentt_df_min: float = field(default=2.1)

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
    compute_loss_in_forward: bool = field(default=True)

    # ---- outputs ----
    return_distribution: bool = field(default=True)
    return_interpretation: bool = field(default=True)
    return_components: bool = field(default=True)
