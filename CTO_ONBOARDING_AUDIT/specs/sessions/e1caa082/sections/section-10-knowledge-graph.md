# Section 10: Alpha Knowledge Graph (AR-3)

## Overview

The system has no structured memory of what it has tested. The research graph can generate hypotheses similar to previously rejected ones because there is no way to query "have we already tested RSI divergence on AAPL in a trending regime?" or "which strategies share the same underlying factors?" The existing `search_knowledge_base` tool (in `src/quantstack/rag/`) searches a text knowledge base with BM25/embedding retrieval, but it has no structured graph relationships between strategies, factors, hypotheses, results, instruments, and regimes.

This section builds a PostgreSQL-native knowledge graph using two new tables (`kg_nodes` and `kg_edges`) with pgvector embeddings for semantic similarity. Four LLM-facing tool functions provide novelty detection, factor overlap analysis, research history queries, and experiment recording. A backfill pipeline populates the graph from existing tables.

## Dependencies

- **section-01-db-migrations**: The `kg_nodes` and `kg_edges` tables, the pgvector extension, and the HNSW index must exist before any knowledge graph code runs. The DB migration section handles table creation in `ensure_schema()` and the Docker image switch to `pgvector/pgvector:pg16`.
- **section-05-event-bus-extensions**: The `EXPERIMENT_COMPLETED` event type is used to trigger knowledge graph updates when overnight autoresearch or weekend parallel research finishes experiments.
- **section-07-overnight-autoresearch**: The `autoresearch_experiments` table provides hypothesis and result data for backfill and ongoing recording.
- **section-08-feature-factory**: Feature candidates feed into factor nodes in the knowledge graph.
- **section-09-weekend-parallel**: Weekend research stream results are recorded as experiments in the knowledge graph.

## Tests First

All tests use pytest with existing fixtures from `tests/unit/conftest.py`.

### Unit Tests (`tests/unit/test_knowledge_graph.py`)

```python
# --- Node / Edge CRUD ---
# Test: create_node inserts node with correct type and properties
# Test: create_edge inserts edge between existing nodes
# Test: create_edge fails gracefully with non-existent node IDs

# --- Novelty Detection ---
# Test: check_hypothesis_novelty returns "redundant" for >0.85 cosine similarity in same regime
# Test: check_hypothesis_novelty returns "novel" for <0.85 similarity
# Test: check_hypothesis_novelty returns "novel" for same hypothesis in different regime

# --- Factor Overlap ---
# Test: check_factor_overlap returns "crowded" when >2 shared factors with existing positions
# Test: check_factor_overlap returns "clear" when <=2 shared factors

# --- Research History ---
# Test: get_research_history returns matching hypotheses by semantic search + regime filter

# --- Experiment Recording ---
# Test: record_experiment creates hypothesis node, result node, factor nodes, and all edges

# --- Temporal Edges ---
# Test: temporal edges (valid_from/valid_to) filter correctly on date queries

# --- Embeddings ---
# Test: embedding generation calls Bedrock Titan (not OpenAI)
# Test: embedding fallback uses local sentence-transformers when Bedrock unavailable
```

### Integration Tests (`tests/integration/test_knowledge_graph.py`)

```python
# Test: population backfill from strategies table creates correct graph structure
# Test: population backfill from ml_experiments creates result nodes with tested_by edges
# Test: factor overlap query with recursive CTE returns correct results for 3-hop traversal
# Test: full lifecycle: record experiment -> check novelty -> detect redundancy on re-test
```

### Performance Tests (`tests/benchmarks/test_knowledge_graph_perf.py`)

```python
# Test: factor overlap query < 100ms with 10K nodes and 50K edges
# Test: novelty detection < 50ms with 10K hypothesis nodes
```

## Database Schema

The two tables are created by section-01-db-migrations in `src/quantstack/db.py`'s `ensure_schema()`. The schema details are included here for reference since this section's code depends on them.

### `kg_nodes`

| Column | Type | Description |
|--------|------|-------------|
| `node_id` | `UUID PRIMARY KEY` | Unique node identifier |
| `node_type` | `TEXT NOT NULL` | One of: `strategy`, `factor`, `hypothesis`, `result`, `instrument`, `regime`, `evidence` |
| `name` | `TEXT NOT NULL` | Human-readable label |
| `properties` | `JSONB DEFAULT '{}'` | Type-specific properties (see node type table below) |
| `embedding` | `vector(1536)` | Semantic embedding for similarity queries |
| `created_at` | `TIMESTAMPTZ DEFAULT now()` | Creation timestamp |
| `updated_at` | `TIMESTAMPTZ DEFAULT now()` | Last update timestamp |

HNSW index on `embedding` column for approximate nearest neighbor search. The `pgvector` extension must be enabled first (`CREATE EXTENSION IF NOT EXISTS vector`).

### `kg_edges`

| Column | Type | Description |
|--------|------|-------------|
| `edge_id` | `UUID PRIMARY KEY` | Unique edge identifier |
| `edge_type` | `TEXT NOT NULL` | One of: `uses`, `tested_by`, `correlates_with`, `contradicted_by`, `favors`, `contains` |
| `source_id` | `UUID REFERENCES kg_nodes(node_id)` | Source node |
| `target_id` | `UUID REFERENCES kg_nodes(node_id)` | Target node |
| `weight` | `FLOAT DEFAULT 1.0` | Edge strength (e.g., correlation magnitude) |
| `properties` | `JSONB DEFAULT '{}'` | Edge-specific metadata |
| `valid_from` | `DATE` | Temporal validity start (nullable for permanent edges) |
| `valid_to` | `DATE` | Temporal validity end (nullable for currently valid edges) |
| `created_at` | `TIMESTAMPTZ DEFAULT now()` | Creation timestamp |

Composite index on `(source_id, edge_type)` and `(target_id, edge_type)` for traversal queries.

### Node Type Properties

| Node Type | Properties Keys | Description |
|-----------|----------------|-------------|
| `strategy` | `status`, `regime_affinity`, `sharpe`, `win_rate` | Links to `strategies` table via `properties.strategy_id` |
| `factor` | `description`, `ic_range`, `decay_rate` | A named predictive feature (e.g., "RSI divergence", "earnings momentum") |
| `hypothesis` | `entry_rules`, `exit_rules`, `test_date`, `outcome` | A tested research idea |
| `result` | `sharpe`, `max_dd`, `ic`, `window` | Backtest or live performance outcome |
| `instrument` | `sector`, `market_cap`, `adv` | A traded symbol |
| `regime` | `trend`, `volatility`, `start_date`, `end_date` | A market state |
| `evidence` | `source`, `confidence`, `date` | Supporting data for a relationship |

### Edge Type Semantics

| Edge Type | Source -> Target | Meaning |
|-----------|-----------------|---------|
| `uses` | strategy -> factor | This strategy uses this factor |
| `tested_by` | hypothesis -> result | This hypothesis produced this result |
| `correlates_with` | factor -> factor | These factors are correlated (weight = magnitude) |
| `contradicted_by` | hypothesis -> evidence | This evidence contradicts this hypothesis |
| `favors` | regime -> strategy | This regime favors this strategy |
| `contains` | instrument -> factor | This instrument exhibits this factor pattern |

Factor correlations change over time, which is why edges have `valid_from`/`valid_to`. A factor that correlated with momentum in 2023 may not in 2025. Temporal edges prevent stale relationships from influencing decisions.

## Implementation

### File: `src/quantstack/knowledge/graph.py` (NEW)

The `KnowledgeGraph` class provides CRUD operations and the four query methods. It uses `db_conn()` context managers for all database access, consistent with the rest of the codebase.

```python
class KnowledgeGraph:
    """PostgreSQL-native knowledge graph for alpha research memory.

    Stores strategies, factors, hypotheses, results, instruments, and regimes
    as nodes with vector embeddings. Edges represent typed relationships with
    optional temporal validity.

    All queries use the existing db_conn() connection pool. No separate
    graph database required -- recursive CTEs handle traversal at QuantStack's
    scale (thousands of nodes, tens of thousands of edges).
    """

    def create_node(self, node_type: str, name: str, properties: dict, embedding: list[float] | None = None) -> str:
        """Insert a node and return its UUID. Idempotent via ON CONFLICT on (node_type, name)."""
        ...

    def create_edge(self, edge_type: str, source_id: str, target_id: str, weight: float = 1.0, properties: dict | None = None, valid_from: str | None = None, valid_to: str | None = None) -> str:
        """Insert an edge. Validates both node IDs exist before insert."""
        ...

    def check_hypothesis_novelty(self, hypothesis_text: str) -> NoveltyResult:
        """Embed the hypothesis, find top-5 similar hypotheses in kg_nodes.

        If any have >0.85 cosine similarity AND were tested in the same regime,
        return "redundant" with links to previous results. Otherwise "novel".

        Uses vector similarity search via pgvector's <=> operator (cosine distance).
        """
        ...

    def check_factor_overlap(self, strategy_id: str) -> OverlapResult:
        """Traverse strategy -> uses -> factor -> uses -> strategy.

        Count shared factors with existing active positions. If >2 shared factors,
        return "crowded" with factor names and affected strategies.

        Uses a recursive CTE limited to 3 hops.
        """
        ...

    def get_research_history(self, topic: str) -> list[HypothesisResult]:
        """Semantic search + edge traversal.

        Example: "What have we learned about momentum in ranging regimes?"
        Finds hypothesis nodes with similar embeddings + regime edges pointing
        to the queried regime.
        """
        ...

    def record_experiment(self, hypothesis: str, result: dict, factors_used: list[str], regime: str) -> str:
        """Create hypothesis node, result node, factor nodes (if new), and all edges.

        Called by the research graph's knowledge_update node and by the
        overnight autoresearch logger.
        """
        ...
```

**Key design decision -- PostgreSQL over Neo4j**: At QuantStack's scale (thousands of nodes, tens of thousands of edges), PostgreSQL is sufficient. Recursive CTEs handle graph traversal for factor overlap (max 3 hops). The operational simplicity of one database far outweighs any query performance difference. No new infrastructure to manage.

### File: `src/quantstack/knowledge/kg_models.py` (NEW)

Pydantic models for knowledge graph I/O. Named `kg_models.py` to avoid collision with the existing `models.py` in the same package.

```python
class NoveltyResult(BaseModel):
    """Result of a hypothesis novelty check."""
    is_novel: bool
    similar_hypotheses: list[SimilarHypothesis]  # top-5 matches with cosine similarity
    recommendation: str  # "novel", "redundant", or "similar_but_different_regime"

class OverlapResult(BaseModel):
    """Result of a factor overlap check."""
    is_crowded: bool
    shared_factor_count: int
    shared_factors: list[str]
    affected_strategies: list[str]

class HypothesisResult(BaseModel):
    """A hypothesis with its test results from the knowledge graph."""
    hypothesis_id: str
    hypothesis_text: str
    test_date: str
    outcome: str
    result_sharpe: float | None
    result_ic: float | None
    regime_at_test: str | None

class SimilarHypothesis(BaseModel):
    """A hypothesis similar to the query, with similarity score."""
    hypothesis_id: str
    name: str
    cosine_similarity: float
    regime: str | None
    outcome: str | None
```

### File: `src/quantstack/knowledge/embeddings.py` (NEW)

Embedding generation with Bedrock Titan primary and local fallback.

```python
def generate_embedding(text: str) -> list[float]:
    """Generate a 1536-dimensional embedding for the given text.

    Primary: Amazon Titan Text Embeddings v2 via Bedrock (the existing
    primary LLM provider). Uses the 'embedding' tier from LLM config.

    Fallback: Local sentence-transformers/all-MiniLM-L6-v2 if Bedrock
    is unavailable (returns 384-dim, zero-padded to 1536 for schema
    compatibility).

    The existing LLM config already defines an 'embedding' tier
    (see src/quantstack/llm/config.py). The current default is
    'ollama/mxbai-embed-large'. This function should resolve the
    embedding model from the active LLM config profile, preferring
    Bedrock Titan when available.
    """
    ...
```

**Embedding dimension note**: The `vector(1536)` column matches Titan v2's output dimension. If the fallback model produces fewer dimensions (e.g., MiniLM produces 384), zero-pad to 1536. Cosine similarity between zero-padded and native vectors is mathematically valid -- the zeros contribute nothing to the dot product. However, cross-model comparisons (some nodes embedded by Titan, others by MiniLM) will have degraded similarity quality. The fallback is for availability, not sustained use.

### File: `src/quantstack/knowledge/population.py` (NEW)

Backfill pipeline to populate the knowledge graph from existing tables.

```python
def backfill_from_strategies() -> int:
    """Create strategy nodes + factor nodes (parsed from entry_rules) + uses edges.

    Reads from the `strategies` table. Parses entry_rules JSON to extract
    factor references (e.g., "RSI > 70" -> factor node "RSI").
    Returns count of nodes created.
    """
    ...

def backfill_from_ml_experiments() -> int:
    """Create result nodes + tested_by edges from the ml_experiments table."""
    ...

def backfill_from_autoresearch() -> int:
    """Create hypothesis nodes + result nodes from autoresearch_experiments table.

    This table is created by section-07-overnight-autoresearch. If the table
    does not exist yet, this function returns 0 gracefully.
    """
    ...

def backfill_from_ic_observations() -> int:
    """Create evidence nodes supporting factor effectiveness from ic_observations."""
    ...

def run_full_backfill() -> dict[str, int]:
    """Run all backfill functions and return counts per source table."""
    ...
```

### File: `src/quantstack/tools/langchain/knowledge_graph_tools.py` (NEW)

LLM-facing tools wrapping the graph query methods. These are registered in `ACTIVE_TOOLS` via `src/quantstack/tools/registry.py`.

```python
@tool
def check_hypothesis_novelty(hypothesis_text: str) -> str:
    """Check if a research hypothesis has already been tested.

    Returns whether the hypothesis is novel or redundant, with links
    to similar previous experiments and their outcomes.
    """
    ...

@tool
def check_factor_overlap(strategy_id: str) -> str:
    """Check if a strategy shares too many factors with existing positions.

    Returns whether the strategy is "crowded" (>2 shared factors) or
    "clear", with details on which factors overlap and which strategies
    are affected.
    """
    ...

@tool
def get_research_history(topic: str) -> str:
    """Query what the system has learned about a research topic.

    Performs semantic search over hypothesis nodes and follows regime
    edges to return relevant past experiments and their outcomes.
    """
    ...

@tool
def record_experiment(hypothesis: str, result_json: str, factors_used: str, regime: str) -> str:
    """Record a completed experiment in the knowledge graph.

    Creates hypothesis, result, and factor nodes with all appropriate edges.
    Called after backtest validation or overnight autoresearch experiments.
    """
    ...
```

### File: `src/quantstack/tools/registry.py` (MODIFY)

Register the four new knowledge graph tools in `ACTIVE_TOOLS`. The tools should be importable from `quantstack.tools.langchain.knowledge_graph_tools` and added to the registry alongside existing tools.

### Integration Points

**Research graph `knowledge_update` node**: After a hypothesis is validated (pass or fail), the research graph should call `record_experiment` to persist the result in the knowledge graph. This is a modification to `src/quantstack/graphs/research/nodes.py` -- add a call to `KnowledgeGraph.record_experiment()` in the node that handles backtest results.

**Research graph `context_load` node**: Before generating new hypotheses, call `check_hypothesis_novelty` to avoid redundant research. This filters the hypothesis queue before validation begins.

**Overnight autoresearch logger**: The overnight runner (section-07) calls `record_experiment` after each experiment completes, feeding the knowledge graph continuously.

**Event bus integration**: When an `EXPERIMENT_COMPLETED` event is received, the supervisor graph can trigger a knowledge graph update if the experiment was not already recorded by the originating graph.

## Key Design Decisions

1. **PostgreSQL over Neo4j**: At QuantStack's scale (thousands of nodes, tens of thousands of edges), PostgreSQL with recursive CTEs handles all traversal patterns needed (max 3 hops for factor overlap). Adding a separate graph database would double infrastructure complexity for negligible query performance improvement. The operational simplicity of one database is the correct tradeoff.

2. **Embeddings for novelty detection**: Text matching would miss semantic equivalents ("RSI divergence" vs. "RSI bearish divergence from price"). Vector similarity catches these via cosine distance. The 0.85 threshold was chosen to balance false positives (declaring novel things redundant) against false negatives (missing true duplicates). It should be tunable via configuration.

3. **Temporal edges with `valid_from`/`valid_to`**: Factor correlations change over time. Without temporal validity, stale relationships influence decisions. The query layer filters edges by current date by default, with an option to query historical relationships for research purposes.

4. **Bedrock Titan for embeddings**: Consistent with the existing LLM provider architecture (Bedrock primary). The `embedding` tier already exists in `src/quantstack/llm/config.py`. The fallback to local sentence-transformers ensures the knowledge graph works even during Bedrock outages, at the cost of degraded similarity quality.

5. **`kg_models.py` naming**: The existing `src/quantstack/knowledge/models.py` contains trade journal and agent state models. Knowledge graph models go in a separate file to avoid a large merge and keep concerns separated.

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `src/quantstack/knowledge/graph.py` | NEW | KnowledgeGraph class with CRUD + 4 query methods |
| `src/quantstack/knowledge/kg_models.py` | NEW | Pydantic models: NoveltyResult, OverlapResult, HypothesisResult, SimilarHypothesis |
| `src/quantstack/knowledge/embeddings.py` | NEW | Embedding generation (Bedrock Titan primary, local fallback) |
| `src/quantstack/knowledge/population.py` | NEW | Backfill from strategies, ml_experiments, autoresearch_experiments, ic_observations |
| `src/quantstack/tools/langchain/knowledge_graph_tools.py` | NEW | 4 LLM-facing @tool functions wrapping graph queries |
| `src/quantstack/tools/registry.py` | MODIFY | Register KG tools in ACTIVE_TOOLS |
| `src/quantstack/graphs/research/nodes.py` | MODIFY | Call record_experiment after hypothesis validation |
| `tests/unit/test_knowledge_graph.py` | NEW | Unit tests for graph CRUD, novelty, overlap, history, recording, embeddings |
| `tests/integration/test_knowledge_graph.py` | NEW | Integration tests for backfill, CTE traversal, full lifecycle |
| `tests/benchmarks/test_knowledge_graph_perf.py` | NEW | Performance tests for query latency at scale |

## Downstream Consumers

- **section-11-consensus-validation**: The bear advocate agent uses `check_factor_overlap` to identify factor crowding as an argument against entry.
- **section-13-meta-agents**: The architecture critic uses knowledge graph query patterns and hit rates to assess research efficiency.
- **section-04-budget-discipline**: The experiment prioritization formula uses `novelty_score` from `check_hypothesis_novelty` (1.0 if novel, 0.1 if redundant) to rank experiments.
- **section-07-overnight-autoresearch**: The overnight runner calls `check_hypothesis_novelty` as a quick-screen step (step 2) before running backtests, and `record_experiment` after each experiment completes.
