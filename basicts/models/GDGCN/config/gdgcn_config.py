from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class GDGCNConfig(BasicTSModelConfig):
    """Configuration for the GDGCN model."""

    input_len: int = field(default=None, metadata={"help": "Historical sequence length (P)."})
    output_len: int = field(default=None, metadata={"help": "Prediction horizon (Q)."})
    num_features: int = field(default=None, metadata={"help": "Number of nodes (N)."})
    input_dim: int = field(default=1, metadata={"help": "Number of input channels per node."})
    residual_channels: int = field(default=32, metadata={"help": "Hidden channels within residual blocks."})
    dilation_channels: int = field(default=32, metadata={"help": "Hidden channels for spatial/temporal blocks."})
    skip_channels: int = field(default=256, metadata={"help": "Channels used for skip connections."})
    kernel_size: int = field(default=2, metadata={"help": "Kernel size for temporal convolutions (if used)."})
    num_layers: int = field(default=8, metadata={"help": "Number of stacked GDGCN layers."})
    spatial_inter_dim: int = field(default=10, metadata={"help": "Intermediate dimension for spatial graph constructor."})
    temporal_inter_dim: int = field(default=4, metadata={"help": "Intermediate dimension for temporal graph constructor."})
    dropout: float = field(default=0.3, metadata={"help": "Dropout rate inside graph convolutions."})
    steps_per_day: int = field(default=288, metadata={"help": "Number of discrete time slots per day."})
    temporal_mode: str = field(default="node_specific",
                               metadata={"help": "Temporal modeling mode: node_specific, tcn, lstm, attention, mlp, mlp_new."})
