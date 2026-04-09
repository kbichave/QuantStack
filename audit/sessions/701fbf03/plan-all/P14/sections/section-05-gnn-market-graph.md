# Section 05: GNN Market Structure Graph

## Objective

Build a graph representation of market structure where nodes are symbols and edges encode correlation, sector membership, and supply chain relationships. This graph is the input for the Graph Attention Network model in section-06.

## Files to Create/Modify

### New Files

- **`src/quantstack/ml/gnn/__init__.py`** — Package init.
- **`src/quantstack/ml/gnn/market_graph.py`** — Graph construction from market data.
- **`src/quantstack/ml/gnn/model.py`** — GAT model definition and training.

### Modified Files

- None (standalone module; signal collector integration is section-06).

## Implementation Details

### `src/quantstack/ml/gnn/market_graph.py`

```
class MarketGraphBuilder:
    """Constructs a market structure graph from price data and metadata."""

    def __init__(
        self,
        correlation_threshold: float = 0.5,
        correlation_window: int = 63,      # ~3 months of trading days
    ):
        ...

    def build(
        self,
        returns: pd.DataFrame,             # columns = symbols, rows = dates
        sector_map: dict[str, str],         # symbol -> GICS sector
        supply_chain: dict[str, list[str]] | None = None,  # symbol -> suppliers/customers
    ) -> MarketGraph:
        """Build graph from returns matrix and metadata.
        
        Edge types:
        1. Correlation edge: rolling correlation > threshold (edge weight = correlation)
        2. Sector edge: same GICS sector (edge weight = 1.0)
        3. Supply chain edge: known supplier/customer relationship (edge weight = 0.8)
        
        Node features (per symbol):
        - Rolling 5/21/63-day returns
        - Rolling 5/21-day volume ratio (vs 63-day avg)
        - RSI(14), MACD histogram
        - Sector one-hot encoding
        
        Returns MarketGraph with adjacency, node features, edge features.
        """

    def to_pyg_data(self, graph: MarketGraph) -> "torch_geometric.data.Data":
        """Convert MarketGraph to PyTorch Geometric Data object for GAT input."""
```

```
@dataclass
class MarketGraph:
    nodes: list[str]                          # symbol names
    node_features: np.ndarray                 # (n_nodes, n_features)
    edge_index: np.ndarray                    # (2, n_edges) — source/target pairs
    edge_features: np.ndarray                 # (n_edges, n_edge_features)
    edge_types: list[str]                     # "correlation", "sector", "supply_chain"
    correlation_matrix: pd.DataFrame          # full correlation matrix for reference
    build_date: date
```

### `src/quantstack/ml/gnn/model.py`

```
class MarketGAT:
    """Graph Attention Network for market structure signals."""

    def __init__(
        self,
        in_channels: int,           # node feature dimension
        hidden_channels: int = 32,
        out_channels: int = 1,      # predict 5-day return per node
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        """Build 2-layer GAT with 4 attention heads."""

    def train_model(
        self,
        graphs: list["torch_geometric.data.Data"],   # walk-forward graph snapshots
        targets: list[np.ndarray],                    # (n_nodes,) 5-day returns per snapshot
        n_epochs: int = 100,
        lr: float = 1e-3,
    ) -> GATTrainResult:
        """Walk-forward training on graph snapshots.
        
        Retrain frequency: monthly (overnight batch).
        Each snapshot is a daily graph with known 5-day forward returns as targets.
        """

    def predict(self, data: "torch_geometric.data.Data") -> GATNodePredictions:
        """Run node-level prediction.
        
        Returns predicted 5-day return for each node (symbol) in the graph,
        plus attention weights between nodes.
        """

    def get_attention_weights(self, data: "torch_geometric.data.Data") -> dict:
        """Extract attention weights from GAT layers.
        
        Returns dict mapping (source_symbol, target_symbol) -> attention_weight.
        High attention weight = model considers this neighbor highly informative.
        Used by contagion signal collector.
        """

    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

```
@dataclass
class GATTrainResult:
    mse: float
    directional_accuracy: float
    n_nodes: int
    n_edges: int
    n_snapshots: int
    checkpoint_path: str

@dataclass
class GATNodePredictions:
    symbols: list[str]
    predicted_returns: np.ndarray         # (n_nodes,)
    attention_weights: dict[tuple[str, str], float]
```

### Key Design Decisions

1. **Correlation threshold 0.5** — empirically, lower thresholds create too-dense graphs; higher misses meaningful relationships.
2. **63-day rolling window** — roughly one quarter, balances stability vs recency.
3. **GAT over GCN** — attention mechanism learns which neighbors matter, producing interpretable attention weights for contagion detection.
4. **Monthly retraining** — graph structure changes slowly; daily would be wasteful.
5. **CPU inference required** — GAT with ~100 nodes and 2 layers is lightweight enough.

## Dependencies

- **PyPI**: `torch`, `torch_geometric` (PyTorch Geometric)
- **Internal**: `quantstack.universe` (symbol list), `quantstack.data.storage.DataStore` (OHLCV), `quantstack.core.features.*` (technical indicators)

## Test Requirements

### `tests/unit/ml/test_gnn_market_graph.py`

1. **Graph construction**: Synthetic returns with known correlation structure — verify edges connect correlated symbols.
2. **Sector edges**: Symbols in same sector have edges; different sectors do not (unless correlated).
3. **Node feature shape**: `node_features.shape == (n_symbols, expected_feature_dim)`.
4. **Edge symmetry**: Correlation edges are undirected (both directions present).
5. **Empty universe**: Handles single-symbol universe gracefully (no edges).

### `tests/unit/ml/test_gnn_model.py`

1. **Synthetic contagion**: Create graph where one node's return perfectly predicts neighbor's — verify GAT learns this.
2. **Attention weight extraction**: Verify `get_attention_weights` returns dict with correct symbol pairs.
3. **Checkpoint save/load**: Train, save, load, predict — verify predictions match.
4. **CPU inference**: Runs without CUDA.

## Acceptance Criteria

- [ ] `MarketGraphBuilder` constructs graphs from returns + sector data + optional supply chain
- [ ] Graph includes correlation, sector, and supply chain edge types
- [ ] `MarketGAT` trains on walk-forward graph snapshots and predicts per-node returns
- [ ] Attention weights are extractable for contagion signal
- [ ] All inference runs on CPU
- [ ] All unit tests pass
