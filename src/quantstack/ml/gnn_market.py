"""Graph Neural Network for market structure modelling.

Builds a graph where nodes are tradeable symbols and edges encode
relationships (return correlation, sector membership, supply-chain links).
A GNN then propagates information across the graph to produce per-symbol
signal scores that capture cross-asset dependencies invisible to
single-symbol models.

Graceful degradation:
  1. torch-geometric available — full GAT-based GNN
  2. torch only — graph convolution via sparse matmul
  3. Neither — spectral clustering + linear model on cluster features
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Optional dependency probing
# ---------------------------------------------------------------------------

_HAS_PYG = False
_HAS_TORCH = False

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    pass

try:
    import torch_geometric  # type: ignore[import-untyped]
    from torch_geometric.nn import GATConv  # type: ignore[import-untyped]

    _HAS_PYG = True
except ImportError:
    pass

logger.info(
    "GNN backend: {}",
    "torch_geometric" if _HAS_PYG else ("torch" if _HAS_TORCH else "spectral_fallback"),
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MarketGraphConfig:
    """Configuration for the market graph and GNN predictor.

    Attributes:
        n_layers: Number of graph attention layers.
        n_heads: Attention heads per layer.
        hidden_dim: Hidden dimension of node embeddings.
        edge_types: Recognised relationship types.
        correlation_threshold: Absolute correlation above which an edge is added.
    """

    n_layers: int = 2
    n_heads: int = 4
    hidden_dim: int = 64
    edge_types: list[str] = field(
        default_factory=lambda: ["correlation", "sector", "supply_chain"],
    )
    correlation_threshold: float = 0.5


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


class MarketGraphBuilder:
    """Construct a graph from return correlations and sector membership."""

    @staticmethod
    def build_graph(
        returns_df: pd.DataFrame,
        sector_map: dict[str, str],
    ) -> dict[str, Any]:
        """Build a market graph.

        Parameters:
            returns_df: DataFrame with symbols as columns and dates as rows.
            sector_map: {symbol: sector_name} mapping.

        Returns:
            {"nodes": list[str],
             "edges": list[tuple[int, int]],
             "edge_features": dict with keys "type" and "weight"}
        """
        symbols = list(returns_df.columns)
        corr = returns_df.corr()

        edges: list[tuple[int, int]] = []
        edge_types: list[str] = []
        edge_weights: list[float] = []

        sym_to_idx = {s: i for i, s in enumerate(symbols)}

        for i, s1 in enumerate(symbols):
            for j, s2 in enumerate(symbols):
                if i >= j:
                    continue

                r = corr.iloc[i, j]
                if abs(r) >= 0.5:  # correlation_threshold applied at call site
                    edges.append((i, j))
                    edge_types.append("correlation")
                    edge_weights.append(float(abs(r)))
                    # Undirected: add reverse
                    edges.append((j, i))
                    edge_types.append("correlation")
                    edge_weights.append(float(abs(r)))

                sect_i = sector_map.get(s1)
                sect_j = sector_map.get(s2)
                if sect_i and sect_j and sect_i == sect_j:
                    # Avoid duplicate if already added via correlation
                    if (i, j) not in edges[-2:]:
                        edges.append((i, j))
                        edge_types.append("sector")
                        edge_weights.append(1.0)
                        edges.append((j, i))
                        edge_types.append("sector")
                        edge_weights.append(1.0)

        logger.info(
            "Market graph: {} nodes, {} edges",
            len(symbols),
            len(edges),
        )

        return {
            "nodes": symbols,
            "edges": edges,
            "edge_features": {"type": edge_types, "weight": edge_weights},
        }


# ---------------------------------------------------------------------------
# GNN predictor
# ---------------------------------------------------------------------------


class GNNPredictor:
    """Per-symbol signal scoring via graph neural network.

    Falls back to spectral-clustering + linear model when torch-geometric
    is not available.
    """

    def __init__(self, config: MarketGraphConfig | None = None) -> None:
        self._cfg = config or MarketGraphConfig()
        self._model: Any = None
        self._fitted = False
        self._node_labels: list[str] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        graph_data: dict[str, Any],
        returns: pd.Series,
    ) -> GNNPredictor:
        """Train on a market graph and corresponding symbol returns.

        Parameters:
            graph_data: Output of MarketGraphBuilder.build_graph().
            returns: Series indexed by symbol with the target signal
                (e.g., forward returns or alpha scores).
        """
        self._node_labels = graph_data["nodes"]

        if _HAS_PYG:
            return self._fit_pyg(graph_data, returns)
        if _HAS_TORCH:
            return self._fit_torch(graph_data, returns)
        return self._fit_spectral(graph_data, returns)

    def _fit_pyg(
        self, graph_data: dict[str, Any], returns: pd.Series
    ) -> GNNPredictor:
        """Full GAT via torch-geometric."""
        from torch_geometric.data import Data  # type: ignore[import-untyped]

        n_nodes = len(graph_data["nodes"])
        # Node features: one-hot identity (simple baseline)
        x = torch.eye(n_nodes)  # type: ignore[name-defined]
        edge_index = torch.tensor(  # type: ignore[name-defined]
            list(zip(*graph_data["edges"])) if graph_data["edges"] else [[], []],
            dtype=torch.long,  # type: ignore[name-defined]
        )
        y = torch.tensor(  # type: ignore[name-defined]
            [returns.get(s, 0.0) for s in graph_data["nodes"]],
            dtype=torch.float32,  # type: ignore[name-defined]
        )

        data = Data(x=x, edge_index=edge_index, y=y)

        model = _GATModel(
            in_dim=n_nodes,
            hidden_dim=self._cfg.hidden_dim,
            n_heads=self._cfg.n_heads,
            n_layers=self._cfg.n_layers,
        )
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)  # type: ignore[name-defined]
        loss_fn = nn.MSELoss()  # type: ignore[name-defined]

        model.train()
        for epoch in range(100):
            optimiser.zero_grad()
            out = model(data.x, data.edge_index)
            loss = loss_fn(out.squeeze(), y)
            loss.backward()
            optimiser.step()

        model.eval()
        self._model = (model, data)
        self._fitted = True
        logger.info("GNN (GAT) trained: {} nodes, loss {:.6f}", n_nodes, loss.item())
        return self

    def _fit_torch(
        self, graph_data: dict[str, Any], returns: pd.Series
    ) -> GNNPredictor:
        """Sparse graph convolution via raw torch (no torch-geometric)."""
        n = len(graph_data["nodes"])
        adj = np.zeros((n, n), dtype=np.float32)
        for i, j in graph_data["edges"]:
            adj[i, j] = 1.0
        # Row-normalise
        row_sum = adj.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        adj_norm = adj / row_sum

        x = np.eye(n, dtype=np.float32)
        y = np.array([returns.get(s, 0.0) for s in graph_data["nodes"]], dtype=np.float32)

        A = torch.tensor(adj_norm)  # type: ignore[name-defined]
        X = torch.tensor(x)  # type: ignore[name-defined]
        Y = torch.tensor(y)  # type: ignore[name-defined]

        W = nn.Linear(n, 1)  # type: ignore[name-defined]
        optimiser = torch.optim.Adam(W.parameters(), lr=1e-3)  # type: ignore[name-defined]
        loss_fn = nn.MSELoss()  # type: ignore[name-defined]

        for _ in range(200):
            optimiser.zero_grad()
            h = A @ X  # one-hop aggregation
            out = W(h).squeeze()
            loss = loss_fn(out, Y)
            loss.backward()
            optimiser.step()

        self._model = (W, A, X)
        self._fitted = True
        logger.info("GNN (sparse torch) trained: {} nodes", n)
        return self

    def _fit_spectral(
        self, graph_data: dict[str, Any], returns: pd.Series
    ) -> GNNPredictor:
        """Spectral clustering + linear regression fallback (numpy only)."""
        from sklearn.cluster import SpectralClustering
        from sklearn.linear_model import Ridge

        n = len(graph_data["nodes"])
        adj = np.zeros((n, n), dtype=np.float32)
        for i, j in graph_data["edges"]:
            adj[i, j] = 1.0

        n_clusters = min(5, max(2, n // 3))
        try:
            sc = SpectralClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                random_state=42,
            )
            labels = sc.fit_predict(adj + adj.T + np.eye(n))
        except Exception:
            labels = np.zeros(n, dtype=int)

        # One-hot cluster features
        features = np.zeros((n, n_clusters), dtype=np.float32)
        for i, lbl in enumerate(labels):
            features[i, lbl] = 1.0

        y = np.array([returns.get(s, 0.0) for s in graph_data["nodes"]], dtype=np.float32)
        model = Ridge(alpha=1.0).fit(features, y)

        self._model = (model, features, labels, n_clusters)
        self._fitted = True
        logger.info("GNN (spectral fallback) trained: {} nodes, {} clusters", n, n_clusters)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, graph_data: dict[str, Any]) -> dict[str, float]:
        """Return per-symbol signal scores.

        Parameters:
            graph_data: Market graph (same schema as build_graph output).

        Returns:
            {symbol: signal_score} dictionary.
        """
        if not self._fitted or self._model is None:
            logger.warning("GNNPredictor not fitted — returning empty scores")
            return {}

        symbols = graph_data["nodes"]

        if _HAS_PYG:
            return self._predict_pyg(symbols)
        if _HAS_TORCH:
            return self._predict_torch(symbols)
        return self._predict_spectral(symbols)

    def _predict_pyg(self, symbols: list[str]) -> dict[str, float]:
        model, data = self._model
        with torch.no_grad():  # type: ignore[name-defined]
            scores = model(data.x, data.edge_index).squeeze().numpy()
        return {s: float(scores[i]) for i, s in enumerate(symbols)}

    def _predict_torch(self, symbols: list[str]) -> dict[str, float]:
        W, A, X = self._model
        with torch.no_grad():  # type: ignore[name-defined]
            h = A @ X
            scores = W(h).squeeze().numpy()
        return {s: float(scores[i]) for i, s in enumerate(symbols)}

    def _predict_spectral(self, symbols: list[str]) -> dict[str, float]:
        model, features, _, _ = self._model
        scores = model.predict(features)
        return {s: float(scores[i]) for i, s in enumerate(symbols)}


# ---------------------------------------------------------------------------
# GAT model (only defined when torch-geometric is available)
# ---------------------------------------------------------------------------

if _HAS_PYG:

    class _GATModel(nn.Module):  # type: ignore[name-defined]
        """Multi-layer Graph Attention Network."""

        def __init__(
            self,
            in_dim: int,
            hidden_dim: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
        ):
            super().__init__()
            self.convs = nn.ModuleList()  # type: ignore[name-defined]
            self.convs.append(GATConv(in_dim, hidden_dim, heads=n_heads, concat=False))
            for _ in range(n_layers - 1):
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=n_heads, concat=False))
            self.head = nn.Linear(hidden_dim, 1)  # type: ignore[name-defined]

        def forward(self, x: Any, edge_index: Any) -> Any:
            for conv in self.convs:
                x = torch.relu(conv(x, edge_index))  # type: ignore[name-defined]
            return self.head(x)
