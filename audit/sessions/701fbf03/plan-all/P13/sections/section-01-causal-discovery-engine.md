# Section 01: Causal Discovery Engine

## Objective

Build the causal graph discovery module that takes a feature matrix and forward returns, runs the PC algorithm via DoWhy, and produces a DAG representing causal relationships. This is the foundation for all downstream causal inference work.

## Dependencies

None (can be implemented in parallel with Section 06).

## Files to Create

### `src/quantstack/core/causal/__init__.py`
- Package init for the causal inference subsystem.
- Expose top-level imports: `CausalGraphBuilder`, `discover_causal_graph`.

### `src/quantstack/core/causal/discovery.py`
- **Class `CausalGraphBuilder`**: stateless builder that accepts a feature DataFrame + forward returns Series.
- **Method `build_graph(features: pd.DataFrame, returns: pd.Series, method: str = "pc") -> CausalGraph`**:
  - Uses DoWhy's `CausalModel` with the PC algorithm (constraint-based) for structure discovery.
  - Input: feature matrix (columns: earnings_revision, insider_buy, short_interest, momentum, volume, etc.) + forward returns column.
  - Output: `CausalGraph` dataclass containing the adjacency list, edge types (directed/undirected), and metadata.
- **Method `validate_against_priors(graph: CausalGraph, priors: dict[str, list[str]]) -> ValidationResult`**:
  - Compare discovered DAG against domain knowledge (known causal paths).
  - Return which expected edges were found, which were missing, and which unexpected edges appeared.
- **Function `discover_causal_graph(features, returns, method="pc") -> CausalGraph`**: module-level convenience wrapper.

### `src/quantstack/core/causal/models.py`
- **Dataclass `CausalGraph`**: adjacency list (dict[str, list[str]]), edge metadata, discovery method, timestamp, validation result.
- **Dataclass `CausalEdge`**: source, target, edge_type (directed/undirected), strength (partial correlation coefficient).
- **Dataclass `ValidationResult`**: confirmed_edges, missing_edges, unexpected_edges, domain_agreement_score (float 0-1).

## Files to Modify

### `src/quantstack/db.py`
- Add `ensure_causal_tables()` call within the existing `ensure_tables()` function.
- Create `causal_graphs` table:
  ```sql
  CREATE TABLE IF NOT EXISTS causal_graphs (
      id SERIAL PRIMARY KEY,
      graph_name TEXT NOT NULL,
      adjacency_list JSONB NOT NULL,
      edge_metadata JSONB,
      discovery_method TEXT NOT NULL DEFAULT 'pc',
      feature_columns TEXT[] NOT NULL,
      num_samples INTEGER NOT NULL,
      domain_agreement_score REAL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
  );
  CREATE INDEX IF NOT EXISTS idx_causal_graphs_name ON causal_graphs(graph_name);
  ```

## Implementation Details

1. **PC Algorithm via DoWhy**: Use `dowhy.gcm` module for causal graph discovery. The PC algorithm is constraint-based: it tests conditional independences to determine edge direction. Suitable for moderate-dimensional feature sets (5-20 features).

2. **Feature-Return Causal Pairs** (priority hypotheses to encode as priors):
   - Earnings revision -> 30-day return (information incorporation)
   - Insider buy -> 60-day return (private information)
   - Short interest change -> 20-day return (supply/demand)
   - Analyst upgrade -> 10-day return (attention/flow)
   - Volume surge -> 5-day return (institutional activity)

3. **Serialization**: Store the DAG as JSONB adjacency list in `causal_graphs` table. The adjacency list format is `{"node_a": ["node_b", "node_c"], ...}` for directed edges from key to values.

4. **Domain Knowledge Validation**: The `priors` dict maps treatment -> expected outcomes (e.g., `{"insider_buy": ["return_60d"], "earnings_revision": ["return_30d"]}`). The validation score is the fraction of expected edges confirmed.

5. **Graceful Degradation**: If DoWhy is not installed, log a warning and return an empty graph rather than crashing. Add `dowhy` and `econml` to optional dependencies.

## Test Requirements

- **Synthetic known-DAG recovery**: Generate data from a known DAG (A -> B -> C, A -> C). Run `build_graph` and verify the discovered graph recovers the correct directed edges.
- **Prior validation**: Provide a graph and priors, verify `validate_against_priors` correctly reports confirmed/missing/unexpected edges.
- **Empty/degenerate input**: Verify graceful handling of empty DataFrame, single-column DataFrame, all-constant columns.
- **Serialization round-trip**: Build a graph, serialize to JSON, deserialize, verify equality.

## Acceptance Criteria

- [ ] `CausalGraphBuilder.build_graph()` returns a valid `CausalGraph` from a feature matrix + returns
- [ ] PC algorithm discovers correct edges on synthetic data with known structure
- [ ] Domain knowledge validation scores match expected values for test cases
- [ ] `causal_graphs` table is created by `ensure_tables()` and graphs can be stored/retrieved
- [ ] All functions handle missing DoWhy gracefully (import guard with warning)
- [ ] Tests pass: `uv run pytest tests/unit/core/causal/test_discovery.py`
