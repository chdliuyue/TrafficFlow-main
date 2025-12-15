from dataclasses import dataclass, field
from typing import Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class STLAformerConfig(BasicTSModelConfig):
    """Configuration for the STLAformer model."""

    input_len: Optional[int] = field(default=None, metadata={"help": "Input sequence length."})
    output_len: Optional[int] = field(default=None, metadata={"help": "Output sequence length for forecasting."})
    num_features: Optional[int] = field(default=None, metadata={"help": "Number of features (sensors)."})
    steps_per_week: int = field(default=2016, metadata={"help": "Number of sampling steps in one week for time-of-week embedding."})
    input_embedding_dim: int = field(default=32, metadata={"help": "Hidden size of input feature projection."})
    tow_embedding_dim: int = field(default=32, metadata={"help": "Hidden size of time-of-week embedding."})
    feed_forward_dim: int = field(default=256, metadata={"help": "Hidden size of the feed-forward network."})
    num_heads: int = field(default=4, metadata={"help": "Number of attention heads."})
    num_layers: int = field(default=6, metadata={"help": "Number of spatial-temporal layers."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})
