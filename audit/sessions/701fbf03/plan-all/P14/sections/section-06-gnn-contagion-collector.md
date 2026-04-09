# Section 06: GNN Contagion Signal Collector

## Objective

Create a signal engine collector that uses GNN attention weights to detect sector contagion — when a highly-connected neighbor drops or rallies significantly, propagate a directional signal to related symbols.

## Dependencies

- **section-05-gnn-market-graph** — requires `MarketGAT`, `MarketGraphBuilder`, `GATNodePredictions`

## Files to Create/Modify

### New Files

- **`src/quantstack/signal_engine/collectors/gnn_contagion.py`** — GNN contagion signal collector.

### Modified Files

- **`src/quantstack/signal_engine/collectors/__init__.py`** — Register the new collector.
- **`src/quantstack/signal_engine/synthesis.py`** — Add `gnn` to weight profiles with weight 0.05.

## Implementation Details

### `src/quantstack/signal_engine/collectors/gnn_contagion.py`

```
_CONTAGION_DROP_THRESHOLD = -0.03    # -3% drop triggers bearish contagion
_CONTAGION_RALLY_THRESHOLD = 0.03    # +3% rally triggers bullish contagion
_ATTENTION_WEIGHT_THRESHOLD = 0.15   # only consider neighbors with attention > this
_GRAPH_CACHE_TTL_HOURS = 4           # rebuild graph at most every 4 hours


async def collect_gnn_contagion(symbol: str, store: DataStore) -> dict[str, Any]:
    """
    Detect contagion risk for *symbol* using GNN attention weights.

    Returns a dict with keys:
        gnn_contagion_score   : float in [-1, 1] — negative = bearish contagion, positive = bullish
        gnn_contagion_sources : list[str] — symbols triggering the contagion signal
        gnn_contagion_type    : str — "bearish_contagion", "bullish_contagion", "none"
        gnn_predicted_return  : float — GNN's predicted 5-day return for this symbol
        gnn_attention_top3    : list[tuple[str, float]] — top 3 most influential neighbors

    Returns {} if no GNN model is available or graph cannot be built.
    """
```

Contagion logic:
1. Load latest GNN model and build/cache the current market graph.
2. Run inference to get attention weights and node predictions.
3. For the target symbol, find neighbors with `attention_weight > 0.15`.
4. Check each high-attention neighbor's recent return:
   - If any neighbor dropped > 3%, compute bearish contagion score weighted by attention.
   - If sector leader (highest market cap in sector) rallied > 3%, compute bullish contagion score.
5. Aggregate: `contagion_score = sum(neighbor_return * attention_weight)` for triggered neighbors.

### Synthesis Integration

Add `gnn` to `_WEIGHT_PROFILES` in `synthesis.py`:
- Weight: 0.05 across all regimes
- Stolen proportionally from existing voters
- Redistributed when GNN signal is unavailable
- IC-based adjustment: reduce to 0.02 if IC < 0.01 after 90 days

### Graph Caching

The market graph is expensive to build (correlation matrix over all symbols). Cache with a 4-hour TTL using a module-level dict keyed by date + hour block. The `MarketGAT` model checkpoint is loaded once and cached for the session.

## Test Requirements

### `tests/unit/signal_engine/test_gnn_contagion.py`

1. **Bearish contagion**: Mock graph where neighbor A (high attention) dropped 5% — verify contagion_type = "bearish_contagion" and score < 0.
2. **Bullish contagion**: Mock graph where sector leader rallied 4% — verify contagion_type = "bullish_contagion" and score > 0.
3. **No contagion**: All neighbors within normal range — verify contagion_type = "none" and score near 0.
4. **Low attention neighbors ignored**: Neighbor dropped 5% but attention weight < threshold — verify no contagion triggered.
5. **No model available**: Returns `{}` without raising.
6. **Never raises**: Inject exception in model load, verify returns `{}`.

## Acceptance Criteria

- [ ] Collector detects bearish contagion from large neighbor drops
- [ ] Collector detects bullish contagion from sector leader rallies
- [ ] Only high-attention neighbors trigger contagion (threshold filtering)
- [ ] Graph and model are cached to avoid redundant computation
- [ ] Registered in collector `__init__.py`
- [ ] Synthesis weights include `gnn` at 0.05
- [ ] Never raises — all failures caught and logged
- [ ] All unit tests pass
