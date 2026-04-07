# Section 02: Context Compaction at Merge Points

**Plan Reference:** Item 5.1
**Dependencies:** None
**Blocks:** None

---

## Problem

Two no-op merge points in the Trading graph pass 65-120KB of uncompacted context per cycle. Reactive pruning at 150K chars drops oldest tool rounds, potentially losing critical data.

## Architecture

**Deterministic compaction** (no LLM call). The Trading graph state is typed — Python code extracts and reshapes fields into Pydantic briefs. Faster, cheaper, more reliable than LLM-based summarization.

Two compaction nodes:
1. `compact_parallel` — after execute_exits + entry_scan/earnings converge → `ParallelMergeBrief`
2. `compact_pre_execution` — after portfolio_review + analyze_options converge → `PreExecutionBrief`

---

## Tests (Write First)

Create `tests/unit/test_compaction.py`:

```python
# --- Brief schema validation ---
# Test: ParallelMergeBrief validates with complete fields
# Test: ParallelMergeBrief rejects signal_strength outside 0.0-1.0
# Test: ParallelMergeBrief accepts empty lists
# Test: PreExecutionBrief validates with complete fields
# Test: PreExecutionBrief rejects missing required fields

# --- Compaction node logic ---
# Test: compact_parallel extracts exits from execute_exits state key
# Test: compact_parallel extracts entries from entry_scan state key
# Test: compact_parallel includes earnings_flags when earnings_analysis ran
# Test: compact_parallel produces empty earnings_flags when earnings skipped
# Test: compact_pre_execution extracts approved/rejected from portfolio_review
# Test: compact_pre_execution extracts options_specs from analyze_options
# Test: compact_pre_execution includes risk_checks dict

# --- Fallback behavior ---
# Test: compaction with malformed state produces degraded brief (not crash)
# Test: degraded brief has compaction_degraded flag set
# Test: compaction with empty state produces valid brief with empty lists

# --- Context size reduction ---
# Test: brief serialization is <20% of raw state for typical cycle data
```

---

## Implementation

### Step 1: Define Brief Schemas

Create `src/quantstack/graphs/trading/briefs.py` with Pydantic BaseModel classes:

- `ExitAction(symbol, action, reason)`
- `EntryCandidate(symbol, signal_strength: Field(ge=0.0, le=1.0), thesis, ewf_bias)`
- `RiskFlag(risk_type, severity, detail)`
- `ParallelMergeBrief(exits, entries, risks, regime, earnings_flags, compaction_degraded=False)`
- `ApprovedEntry(symbol, position_size, structure, rationale)`
- `OptionsSpec(symbol, legs, max_loss, target_profit)`
- `PreExecutionBrief(approved, rejected, options_specs, risk_checks, compaction_degraded=False)`

### Step 2: Implement Compaction Nodes

Create `src/quantstack/graphs/trading/compaction.py`:

- `compact_parallel(state)` — Extract exits, entries, earnings flags, risks, regime. Wrap in try/except; on failure return degraded brief.
- `compact_pre_execution(state)` — Extract approved/rejected, options specs, risk checks. Same fallback pattern.

### Step 3: Update Graph State Schema

Add `parallel_brief` and `pre_execution_brief` keys to Trading graph state. Keep raw state keys for checkpoint debugging.

### Step 4: Update Graph Wiring

Replace `merge_parallel` → `compact_parallel` and `merge_pre_execution` → `compact_pre_execution` in `graph.py`.

### Step 5: Update Downstream Agents

- `risk_sizing` — read from `parallel_brief`
- `execute_entries` — read from `pre_execution_brief`, update system prompt
- `reflect` — receive both brief and execution outcomes
- Serialize briefs via `.model_dump_json()` in agent context

### Step 6: Measure Context Reduction

Log raw state size vs brief size at `execute_entries` entry. Target: 40%+ reduction.

---

## Rollback

Restore no-op merges; raw state keys still in checkpoint. No data loss risk.

## Files

| File | Change |
|------|--------|
| `src/quantstack/graphs/trading/briefs.py` | **Create** — Pydantic schemas |
| `src/quantstack/graphs/trading/compaction.py` | **Create** — Compaction logic |
| `src/quantstack/graphs/trading/graph.py` | **Modify** — Replace merge nodes |
| `src/quantstack/graphs/trading/nodes.py` | **Modify** — Update risk_sizing |
| `src/quantstack/graphs/trading/config/agents.yaml` | **Modify** — executor prompt |
| `src/quantstack/graphs/agent_executor.py` | **Modify** — Brief serialization |
| `tests/unit/test_compaction.py` | **Create** |
