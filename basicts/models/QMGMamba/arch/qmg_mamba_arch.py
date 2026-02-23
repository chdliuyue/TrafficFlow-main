import torch
from torch import nn

from ..config.qmg_mamba_config import QMGMambaConfig
from .layers import (ForecastHead, QMGAdjacency, QMGConvBlock,
                     TimestampEmbedding, TemporalBackbone)


class QMGMamba(nn.Module):
    """QMGMamba: dynamic (quantum-magnetic) graph learning + swappable temporal backbone.

    Target use-case:
        - PeMS03/04/07/08 traffic flow prediction
        - No trusted/static adjacency matrix available
        - Need a learned directed graph that can express:
            * state-dependent structure (dynamic)
            * negative correlations (optional)
            * propagation direction / time lag (phase)
        - Long-range temporal dependencies captured by Mamba2-like selective SSM (or Transformer/RNN)

    Input/Output (BasicTS convention):
        inputs:             [B, input_len, num_features]
        targets:            [B, output_len, num_features]
        inputs_timestamps:  [B, input_len, num_timestamps]
        targets_timestamps: [B, output_len, num_timestamps]

    Output:
        prediction:         [B, output_len, num_features]

    If config.output_attentions or config.return_graph:
        returns a dict with extra graph tensors for visualization.
    """

    def __init__(self, config: QMGMambaConfig):
        super().__init__()
        self.config = config

        self.input_len = int(config.input_len)
        self.output_len = int(config.output_len)
        self.num_features = int(config.num_features)
        self.in_channels = int(getattr(config, "in_channels", 1))

        self.hidden_size = int(config.hidden_size)
        self.dropout = float(config.dropout)

        self.use_timestamp = bool(config.use_timestamp)
        self.timestamp_sizes = config.timestamp_sizes

        self.use_omg = bool(config.use_omg)
        self.omg_dynamic = bool(config.omg_dynamic)
        self.forbid_negative_edges = bool(config.forbid_negative_edges)
        self.use_complex_graph = bool(config.use_complex_graph)
        self.return_graph = bool(getattr(config, "return_graph", False))
        self.output_attentions = bool(getattr(config, "output_attentions", False))

        # ====== embeddings ======
        # value embedding: per node scalar -> hidden
        self.value_embedding = nn.Linear(self.in_channels, self.hidden_size)

        # node id embedding (helps to disambiguate sensors)
        self.node_embedding = nn.Parameter(torch.randn(self.num_features, self.hidden_size) * 0.02)

        # timestamp embeddings (shared for input + future)
        self.ts_embedding = None
        if self.use_timestamp:
            self.ts_embedding = TimestampEmbedding(self.timestamp_sizes, self.hidden_size, dropout=self.dropout)

        self.input_norm = nn.LayerNorm(self.hidden_size)
        self.embed_dropout = nn.Dropout(self.dropout)

        # ====== QMG / OMG spatial module ======
        self.graph_blocks = nn.ModuleList([])
        self._last_graph_cache = None  # optional debug

        if self.use_omg:
            adjacency = QMGAdjacency(
                num_nodes=self.num_features,
                d_emb=int(config.graph_emb_dim),
                ctx_dim=self.hidden_size,
                dynamic=self.omg_dynamic,
                forbid_negative_edges=self.forbid_negative_edges,
                use_complex=self.use_complex_graph,
                temperature=float(config.graph_temperature),
                topk=int(config.graph_topk),
                add_self_loops=bool(config.add_self_loops),
                self_loop_weight=float(config.self_loop_weight),
                dropout=self.dropout,
            )

            for _ in range(int(config.num_qmg_layers)):
                self.graph_blocks.append(
                    QMGConvBlock(
                        d_model=self.hidden_size,
                        adjacency=adjacency,
                        dropout=self.dropout,
                        activation="silu",
                    )
                )

        # ====== Temporal backbone ======
        self.temporal = TemporalBackbone(
            backbone=str(config.temporal_backbone),
            d_model=self.hidden_size,
            num_layers=int(config.num_temporal_layers),
            dropout=self.dropout,
            n_heads=int(config.n_heads),
            d_ff=int(config.intermediate_size),
            transformer_norm_first=bool(config.transformer_norm_first),
            mamba_d_state=int(config.mamba_d_state),
            mamba_d_conv=int(config.mamba_d_conv),
            rnn_bidirectional=bool(config.rnn_bidirectional),
        )

        # ====== Forecast head ======
        self.head = ForecastHead(
            d_model=self.hidden_size,
            horizon=self.output_len,
            dropout=self.dropout,
            use_future_timestamp=bool(config.use_future_timestamp),
            timestamp_embedder=self.ts_embedding,
        )

    def _build_node_context(self, h: torch.Tensor) -> torch.Tensor:
        """Build node context for dynamic graph.

        Args:
            h: [B, L, N, D]

        Returns:
            node_ctx: [B, N, D]
        """
        if getattr(self.config, "omg_context", "last") == "mean":
            return h.mean(dim=1)
        return h[:, -1, :, :]

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        inputs_timestamps: torch.Tensor,
        targets_timestamps: torch.Tensor,
    ):
        """Forward."""
        # Support both:
        #   - [B, L, N]          (most BasicTS forecasting models)
        #   - [B, L, N, C]       (some spatio-temporal datasets)
        if inputs.dim() == 3:
            B, L, N = inputs.shape
            x = inputs.unsqueeze(-1)  # [B, L, N, 1]
        elif inputs.dim() == 4:
            B, L, N, C = inputs.shape
            x = inputs  # [B, L, N, C]
            if C != self.in_channels:
                raise ValueError(f"config.in_channels={self.in_channels} but inputs has C={C}")
        else:
            raise ValueError(
                f"Expected inputs with shape [B, L, N] or [B, L, N, C], got {tuple(inputs.shape)}"
            )

        if N != self.num_features:
            raise ValueError(
                f"config.num_features={self.num_features} but inputs node dim is {N}. "
                "For PeMS, num_features should equal num_sensors."
            )

        h = self.value_embedding(x)  # [B, L, N, D]

        # add node embedding
        h = h + self.node_embedding.unsqueeze(0).unsqueeze(0)

        # add timestamp embedding (broadcast to nodes)
        if self.ts_embedding is not None and inputs_timestamps is not None:
            t_in = self.ts_embedding(inputs_timestamps)  # [B, L, D]
            h = h + t_in.unsqueeze(2)

        h = self.embed_dropout(self.input_norm(h))

        # ====== spatial graph propagation ======
        graph_info = None
        if self.use_omg and len(self.graph_blocks) > 0:
            node_ctx = self._build_node_context(h) if self.omg_dynamic else None

            # stack blocks; return last graph if needed
            last_graph = None
            for blk in self.graph_blocks:
                h, g = blk(h, node_ctx=node_ctx, return_graph=(self.return_graph or self.output_attentions))
                if g is not None:
                    last_graph = g

            graph_info = last_graph
            self._last_graph_cache = last_graph

        # ====== temporal modeling (per node) ======
        h = self.temporal(h)  # [B, L, N, D]

        # use last hidden per node
        h_last = h[:, -1, :, :]  # [B, N, D]

        # horizon-wise decoding; uses future timestamps if enabled
        pred = self.head(h_last, targets_timestamps)  # [B, H, N]

        if self.output_attentions or self.return_graph:
            return {
                "prediction": pred,
                "graph": graph_info,
            }
        return pred
