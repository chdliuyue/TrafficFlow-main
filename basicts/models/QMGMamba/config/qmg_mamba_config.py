"""QMGMamba configuration.

This config follows the same style as BasicTS models (see AutoformerConfig).
All important hyper-parameters are exposed here for easy ablations:
- temporal backbone switch (mamba2 / transformer / gru / lstm)
- graph module switch (use_omg)
- static vs dynamic graph (omg_dynamic)
- allow vs forbid negative edges (forbid_negative_edges)

Note:
- `num_features` in BasicTS long-term forecasting convention usually means the
  number of variables. For PeMS traffic forecasting in BasicTS, we typically
  set `num_features == num_nodes` (number of sensors).
"""

from dataclasses import dataclass, field
from typing import Sequence, Optional
from basicts.configs import BasicTSModelConfig



@dataclass
class QMGMambaConfig(BasicTSModelConfig):
    """Config class for QMGMamba model."""

    # ====== data / io ======
    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of features (typically num_sensors for PeMS)."})

    # Some BasicTS datasets store values as [B, L, N] (N sensors). In that case,
    # each sensor has 1 channel. Keep `in_channels=1` unless you have multi-variate features.
    in_channels: int = field(default=1, metadata={"help": "Input channels per feature (per node)."})

    # ====== embeddings ======
    hidden_size: int = field(default=128, metadata={"help": "Hidden size / model dimension."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})

    use_timestamp: bool = field(default=True, metadata={"help": "Whether to use timestamp categorical embeddings."})
    # Typical for PeMS: [time_of_day(288), day_of_week(7)]
    timestamp_sizes: Optional[Sequence[int]] = field(default=None, metadata={"help": "Vocabulary sizes for each timestamp field."})

    # Whether to use *future* timestamps in the forecasting head (recommended).
    use_future_timestamp: bool = field(default=True, metadata={"help": "Whether to condition forecasting head on targets_timestamps."})

    # ====== OMG/QMG graph module ======
    # In the earlier discussion you used the name OMG; here we keep the same term for toggles.
    use_omg: bool = field(default=True, metadata={"help": "Enable the QMG/OMG graph module."})
    omg_dynamic: bool = field(default=True, metadata={"help": "Dynamic graph (state-dependent) vs static graph."})

    # If True, edges are constrained to be non-negative; if False, signed edges are allowed.
    forbid_negative_edges: bool = field(default=False, metadata={"help": "Whether to forbid negative edges in learned graph."})

    # Complex (phase) graph: magnitude + phase; phase encodes propagation direction / time-lag.
    use_complex_graph: bool = field(default=True, metadata={"help": "Whether to use complex weights (magnitude+phase)."})

    graph_emb_dim: int = field(default=32, metadata={"help": "Embedding dimension used to parameterize the learned graph."})
    num_qmg_layers: int = field(default=2, metadata={"help": "Number of stacked QMG conv layers."})

    graph_temperature: float = field(default=1.0, metadata={"help": "Temperature for edge logits."})
    graph_topk: int = field(default=20, metadata={"help": "Top-k sparsification per node (<=0 means no sparsification)."})
    add_self_loops: bool = field(default=True, metadata={"help": "Whether to add self-loops to the learned graph."})
    self_loop_weight: float = field(default=0.2, metadata={"help": "Weight of self-loop when add_self_loops=True."})

    # context used to build the dynamic graph
    omg_context: str = field(default="last", metadata={"help": "Context for dynamic graph: 'last' or 'mean'."})

    # Optional regularization knobs (can be used in the trainer/loss if needed)
    # We keep them here for completeness; you can choose to ignore them.
    graph_l1: float = field(default=0.0, metadata={"help": "Optional L1 regularization weight for adjacency magnitude."})
    graph_entropy: float = field(default=0.0, metadata={"help": "Optional entropy regularization for adjacency."})

    # ====== temporal backbone ======
    # Switchable temporal module.
    # - 'mamba2': selective state-space scan (implemented in pure PyTorch)
    # - 'transformer': TransformerEncoder
    # - 'gru'/'lstm'
    temporal_backbone: str = field(default="gru", metadata={"help": "Temporal backbone: mamba2 | transformer | gru | lstm"})
    num_temporal_layers: int = field(default=4, metadata={"help": "Number of temporal layers."})

    # Transformer params
    n_heads: int = field(default=8, metadata={"help": "Transformer attention heads."})
    intermediate_size: int = field(default=256, metadata={"help": "Transformer FFN intermediate size."})
    transformer_norm_first: bool = field(default=True, metadata={"help": "Use pre-norm transformer encoder."})

    # RNN params
    rnn_bidirectional: bool = field(default=False, metadata={"help": "Use bidirectional RNN (usually False for forecasting)."})

    # Mamba2 (SSM) params
    mamba_d_state: int = field(default=16, metadata={"help": "State dimension per channel in selective SSM."})
    mamba_d_conv: int = field(default=4, metadata={"help": "Depthwise conv kernel size (local mixing)."})

    # ====== output / debugging ======
    output_attentions: bool = field(default=False, metadata={"help": "Whether to output graph/attention weights for visualization."})
    return_graph: bool = field(default=False, metadata={"help": "Return learned graph along with prediction."})
