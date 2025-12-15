from torch import nn

from .attention import InLineAttention
from .fusion import CGAFusion


class STLAEncoderLayer(nn.Module):
    """Spatial-Temporal Linear Attention encoder layer."""

    def __init__(self, model_dim: int, feed_forward_dim: int = 2048, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.attns = InLineAttention(model_dim, num_heads)
        self.attnt = InLineAttention(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.lnt = nn.LayerNorm(model_dim)
        self.lns = nn.LayerNorm(model_dim)
        self.ln = nn.LayerNorm(model_dim)
        self.dropoutt = nn.Dropout(dropout)
        self.dropouts = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.cgf = CGAFusion(model_dim)

    def forward(self, x, node_emb, time_emb):
        xs = (x + node_emb).transpose(2, -2)
        xt = (x + time_emb).transpose(1, -2)

        residuals = xs
        residualt = xt

        outt = self.attnt(xt)
        outs = self.attns(xs)

        outs = self.dropouts(outs)
        outt = self.dropoutt(outt)
        outs = self.lns(residuals + outs)
        outt = self.lnt(residualt + outt)

        outs = outs.transpose(2, -2)
        outt = outt.transpose(1, -2)

        out = self.cgf(outs, outt)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout(out)
        out = self.ln(residual + out)

        return out
