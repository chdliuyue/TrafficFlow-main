from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class GRUForecastConfig(BasicTSModelConfig):
    """Configuration for the GRUForecast model."""

    input_len: int = field(default=None, metadata={"help": "Length of input history sequences."})
    output_len: int = field(default=None, metadata={"help": "Number of time steps to predict."})
    num_features: int = field(default=1, metadata={"help": "Number of features in each time step."})
    hidden_size: int = field(default=64, metadata={"help": "Hidden size of the GRU encoder."})
    num_layers: int = field(default=2, metadata={"help": "Number of stacked GRU layers."})
    dropout: float = field(default=0.0, metadata={"help": "Dropout applied between GRU layers."})
    bidirectional: bool = field(default=False, metadata={"help": "Whether to use a bidirectional GRU."})
