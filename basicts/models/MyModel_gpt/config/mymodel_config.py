
from dataclasses import dataclass, field
from typing import Sequence

from basicts.configs import BasicTSModelConfig


@dataclass
class MyModelConfig(BasicTSModelConfig):
    """
    Core architecture config for traffic flow forecasting:

    Backbone (mid-layer) feature H  -> 3 parallel branches
      1) H itself (base branch)
      2) Spatial explainable branch (low-rank relation, avoids N×N)
      3) Temporal explainable branch (novel spectral-token attention)

    Then convex residual fusion (stable, interpretable), and a statistical distribution head
    with NLL loss (Part C: probabilistic, distribution-aware forecasting).

    Shapes:
      inputs:             [B, L, N]
      targets:            [B, O, N]
      inputs_timestamps:  [B, L, T]   (normalized to [0,1])
      targets_timestamps: [B, O, T]   (normalized to [0,1])

    Notes for traffic engineers:
      - spatial branch learns low-rank network co-movement modes (r << N), not an explicit adjacency.
      - temporal branch attends over *periodic spectral tokens* (daily/weekly harmonics), giving
        interpretable weights over periodicities.
      - distribution head models heteroscedastic uncertainty (per node, per horizon), e.g. Student-t.
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
    backbone_tap_layer: int = field(
        default=-1,
        metadata={"help": "which layer's final hidden to branch from; -1 means last layer"},
    )

    # input conditioning
    use_input_timestamps: bool = field(default=True)

    # transformer-only knobs (only used if backbone_type='transformer')
    transformer_nhead: int = field(default=4)
    transformer_ffn_ratio: float = field(default=4.0)
    transformer_norm_first: bool = field(default=True)
    transformer_use_positional_encoding: bool = field(default=True)

    # ============================================================
    # Identity embeddings (strong for large-N traffic networks)
    # ============================================================
    node_emb_dim: int = field(default=64)   # set 0 to disable
    step_emb_dim: int = field(default=32)   # set 0 to disable
    dropout: float = field(default=0.1)     # shared dropout for embeddings/decoder

    # ============================================================
    # Branch 1) Spatial explainability: low-rank relation (avoid N×N)
    #   A(b,o) = B diag(s(b,o)) B^T   (rank=r, s>=0)
    #   compute AH without forming N×N:  B ( s ⊙ (B^T H) )
    # ============================================================
    enable_spatial: bool = field(default=True)
    spatial_rank: int = field(default=64)            # r
    spatial_alpha: float = field(default=0.1)        # residual strength
    spatial_scale_hidden: int = field(default=256)
    spatial_scale_dropout: float = field(default=0.1)
    reg_spatial_orth: float = field(default=1e-4)    # stabilize signed/no-normalize regime
    spatial_use_output_timestamps: bool = field(default=True)

    # ============================================================
    # Branch 2) Temporal explainability (NEW): Spectral-Token Attention
    #   - tokens = {daily harmonics} U {weekly harmonics}
    #   - attention weights are interpretable contributions of periodicities
    # ============================================================
    enable_time: bool = field(default=True)
    time_tod_harmonics: int = field(default=4)
    time_dow_harmonics: int = field(default=2)
    time_attn_dim: int = field(default=64)
    time_alpha: float = field(default=1.0)            # residual strength
    time_gate_bound: float = field(default=1.0)       # gate is tanh(.) * bound

    # ============================================================
    # Convex residual fusion (stable):
    #   F = w0*H_base + ws*H_spatial + wt*H_time
    #   w0,ws,wt>0, w0+ws+wt=1  (convex constraint)
    # ============================================================
    fusion_learnable: bool = field(default=True)
    fusion_raw_spatial_init: float = field(default=-1.0)
    fusion_raw_time_init: float = field(default=-1.0)

    # ============================================================
    # Decoder + optional linear skip (strong baseline)
    # ============================================================
    decoder_mlp_hidden: int = field(default=256)       # 0 -> linear
    decoder_use_output_timestamps: bool = field(default=True)
    enable_linear_skip: bool = field(default=True)

    # ============================================================
    # Distribution head + statistical loss (Part C)
    # ============================================================
    likelihood: str = field(
        default="studentt",
        metadata={"help": "none|gaussian|studentt|quantile"},
    )
    min_scale: float = field(default=0.01)            # prevents NLL explosion when scale -> 0

    # student-t df
    studentt_df_mode: str = field(default="learned_global", metadata={"help": "fixed|learned_global"})
    studentt_df_init: float = field(default=10.0)
    studentt_df_min: float = field(default=2.1)

    # quantile regression
    quantiles: Sequence[float] = field(default=(0.1, 0.5, 0.9))

    # loss weights
    point_loss: str = field(default="mae", metadata={"help": "mae|mse|huber"})
    huber_delta: float = field(default=1.0)
    lambda_point: float = field(default=1.0)
    lambda_nll: float = field(default=0.1)

    # runner compatibility
    compute_loss_in_forward: bool = field(default=True)

    # outputs (debug / paper analysis)
    return_interpretation: bool = field(default=False)
    return_components: bool = field(default=False)
