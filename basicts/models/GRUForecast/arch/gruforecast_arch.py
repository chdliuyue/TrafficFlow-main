import torch
from torch import nn

from ..config.gruforecast_config import GRUForecastConfig


class GRUForecast(nn.Module):
    """A simple GRU-based multi-step forecaster."""

    def __init__(self, config: GRUForecastConfig):
        super().__init__()

        self.num_features = config.num_features
        self.output_len = config.output_len
        hidden_size = config.hidden_size
        num_directions = 2 if config.bidirectional else 1

        self.encoder = nn.GRU(
            input_size=config.num_features,
            hidden_size=hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_size * num_directions, hidden_size * num_directions),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size * num_directions, config.output_len * config.num_features),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs (torch.Tensor): Input tensor shaped [batch, input_len, num_features].

        Returns:
            torch.Tensor: Forecast shaped [batch, output_len, num_features].
        """

        _, hidden = self.encoder(inputs)
        last_hidden = hidden[-1]
        forecast = self.projection(last_hidden)
        return forecast.view(-1, self.output_len, self.num_features)
