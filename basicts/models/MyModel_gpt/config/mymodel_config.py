
from dataclasses import dataclass, field
from typing import Sequence

from basicts.configs import BasicTSModelConfig


@dataclass
class MyModelConfig(BasicTSModelConfig):
    """
    MyModelConfig (Student-t only, with ablation switches)

    Core design:
      Backbone -> mid-layer feature H -> 3 parallel residual branches
        (1) Base: H
        (2) Spatial explainability: low-rank interaction, avoids N×N
        (3) Temporal explainability: Spectral-Token Attention (periodic harmonic tokens)

      Convex fusion:
        F = w0*H + ws*H_spatial + wt*H_time,  w0,ws,wt>0, w0+ws+wt=1

      Distributional forecasting (Student-t, innovation #3):
        y|x ~ StudentT(mu(x), sigma(x), nu(x))
        - mu is the point forecast
        - sigma/nu are learned by neural heads (heteroscedastic + heavy-tail)
        - NLL provides gradients to learn sigma/nu for incident/weather long-tail uncertainty.

    Ablations (paper-friendly):
      - enable_spatial: ablate innovation #1
      - enable_time: ablate innovation #2
      - enable_distribution: ablate innovation #3 (disables Student-t NLL & param heads)

    Shapes:
      inputs:             [B, L, N]
      targets:            [B, O, N]
      inputs_timestamps:  [B, L, T]   (normalized to [0,1])
      targets_timestamps: [B, O, T]   (normalized to [0,1])
    """

    # ---- required ----
    input_len: int = field(default=None)        # L
    output_len: int = field(default=None)       # O
    num_features: int = field(default=None)     # N
    num_timestamps: int = field(default=2)      # T (e.g., [tod_norm, dow_norm])
    timestamp_sizes: Sequence[int] = field(default=(288, 7))  # meta only

    # ---- preprocessing ----
    last_value_centering: bool = field(default=True)

    # ============================================================
    # Backbone (flexible): GRU / LSTM / Transformer
    # ============================================================
    backbone_type: str = field(default="gru", metadata={"help": "gru|lstm|transformer"})
    backbone_hidden_size: int = field(default=256)   # D
    backbone_layers: int = field(default=2)
    backbone_dropout: float = field(default=0.1)
    backbone_tap_layer: int = field(default=-1)      # -1 means last layer

    use_input_timestamps: bool = field(default=True)

    # transformer-only knobs
    transformer_nhead: int = field(default=4)
    transformer_ffn_ratio: float = field(default=4.0)
    transformer_norm_first: bool = field(default=True)
    transformer_use_positional_encoding: bool = field(default=True)

    # ============================================================
    # Identity embeddings (help large-N networks)
    # ============================================================
    node_emb_dim: int = field(default=64)   # 0 to disable
    step_emb_dim: int = field(default=32)   # 0 to disable
    dropout: float = field(default=0.1)

    # ============================================================
    # Ablation switch #1: Spatial branch
    # ============================================================
    enable_spatial: bool = field(default=True)
    spatial_rank: int = field(default=64)
    spatial_alpha: float = field(default=0.1)
    spatial_scale_hidden: int = field(default=256)
    spatial_scale_dropout: float = field(default=0.1)
    reg_spatial_orth: float = field(default=1e-4)
    spatial_use_output_timestamps: bool = field(default=True)

    # ============================================================
    # Ablation switch #2: Time branch
    # ============================================================
    enable_time: bool = field(default=True)
    time_tod_harmonics: int = field(default=4)
    time_dow_harmonics: int = field(default=2)
    time_attn_dim: int = field(default=64)
    time_alpha: float = field(default=1.0)
    time_gate_bound: float = field(default=1.0)

    # ============================================================
    # Convex fusion (stable): base + spatial + time
    # ============================================================
    fusion_learnable: bool = field(default=True)
    fusion_raw_spatial_init: float = field(default=-1.0)
    fusion_raw_time_init: float = field(default=-1.0)

    # ============================================================
    # Ablation switch #3: Distributional fitting (Student-t)
    # ============================================================
    enable_distribution: bool = field(default=True)

    dist_trunk_hidden: int = field(default=256)   # shared trunk width for mu & params heads (0 -> linear)
    dist_trunk_layers: int = field(default=2)     # keep small (1~2) for stability

    min_scale: float = field(default=0.01)        # sigma floor for numeric stability

    # df learning mode: "learned_from_features" (recommended) or "learned_global"
    studentt_df_mode: str = field(default="learned_from_features")
    studentt_df_init: float = field(default=10.0)   # used as bias init
    studentt_df_min: float = field(default=2.1)
    studentt_df_max: float = field(default=60.0)

    # ============================================================
    # Decoder conditioning
    # ============================================================
    decoder_use_output_timestamps: bool = field(default=True)
    enable_linear_skip: bool = field(default=True)

    # ============================================================
    # Training loss (inside forward for runner compatibility)
    # ============================================================
    point_loss: str = field(default="mae", metadata={"help": "mae|mse|huber"})
    huber_delta: float = field(default=1.0)
    lambda_point: float = field(default=1.0)

    # NLL weight (effective only when enable_distribution=True)
    lambda_nll: float = field(default=0.02)

    compute_loss_in_forward: bool = field(default=True)

    # outputs (debug / paper analysis)
    return_interpretation: bool = field(default=False)
    return_components: bool = field(default=False)
