# Section 4: Output Schema Validation with Retry

## Background

QuantStack runs 21 LLM agents across three LangGraph StateGraphs (trading, research, supervisor). Every agent's raw text output passes through `parse_json_response()` in `src/quantstack/graphs/agent_executor.py` (lines 474-521). This function tries direct JSON parse, then substring extraction, then returns an empty fallback dict. There is no schema validation, no retry on malformed output, no logging beyond debug level, and no dead letter queue. A parse failure is silently swallowed and replaced with an untyped fallback.

The most dangerous consequence: the `safety_check` agent's fallback is `{"halted": False}`, meaning a parse failure **bypasses the safety check entirely**. This is a P0 safety inversion -- failure should halt, not proceed.

This section has **no dependencies** on other sections and can be implemented in parallel with everything else.

## Tests (Write These First)

### Fallback audit tests

These tests enforce that every safety-critical agent fails CLOSED, not OPEN.

```python
# Test: safety_check fallback is {"halted": True} (fail-CLOSED, not fail-OPEN)
# Test: risk assessment fallback rejects (not approves)
# Test: entry_scan fallback is [] (no entries — safe)
# Test: ALL 21 agent fallbacks documented and classified as fail-safe
```

The fallback audit test should enumerate all 21 agents and assert each fallback value matches the expected fail-safe default. This is a table-driven test: a dict mapping agent name to expected fallback, iterated with `pytest.mark.parametrize`.

### Pydantic model tests

```python
# Test: MarketIntelOutput validates known-good agent output sample
# Test: MarketIntelOutput rejects output missing required fields
# Test: EntrySignalOutput validates known-good sample
# Test: (repeat for each of 21 models with representative samples)
# Test: each model's JSON schema is serializable (for retry prompt inclusion)
```

Each model test should use a realistic sample output captured from actual agent runs (store samples as fixtures or inline dicts). The serialization test verifies `model.model_json_schema()` returns valid JSON -- this is required for the retry prompt.

### parse_and_validate() tests

```python
# Test: valid JSON + valid schema → parsed and validated correctly
# Test: valid JSON + invalid schema → retry triggered with schema hint
# Test: invalid JSON → retry triggered
# Test: retry succeeds → result returned, flagged as retried in audit trail
# Test: retry fails → dead letter queue entry created, fail-safe fallback returned
# Test: retried output has "retried" flag in audit trail
```

These tests mock the LLM call to control what the retry returns. The "retried" flag test verifies that downstream consumers can distinguish first-pass outputs from retry-path outputs.

### Dead letter queue tests

```python
# Test: agent_dead_letters table created with correct schema
# Test: DLQ entry contains agent_name, cycle_id, raw_output, parse_error, retry_attempted
# Test: DLQ queryable by agent_name and time range
# Test: supervisor health check flags agent with >10% DLQ rate
```

The DLQ table test should verify the schema via an integration test against a real (or test) PostgreSQL instance. The supervisor health check test should verify that when DLQ entries exceed 10% of cycles for any agent over the last 24 hours, a warning is raised.

## Implementation

### Step 1: Define 21 Pydantic output models

**File:** `src/quantstack/graphs/schemas/` (new directory) or extend `src/quantstack/tools/models.py` if it remains manageable.

Create one Pydantic `BaseModel` per agent. Each model captures the exact output shape that agent is expected to produce. Two representative examples:

```python
class MarketIntelOutput(BaseModel):
    """Output schema for market_intel agent."""
    headlines: list[str]
    risk_alerts: list[str]
    event_calendar: list[dict]
    sector_news: dict
    sentiment: Literal["bullish", "neutral", "bearish"]

class EntrySignalOutput(BaseModel):
    """Output schema for entry_scan agent."""
    signals: list[SignalCandidate]
    reasoning: str
    regime_assessment: str
```

To build the full set of 21 models, audit each agent's config in:
- `src/quantstack/graphs/research/config/agents.yaml` (8 agents)
- `src/quantstack/graphs/trading/config/agents.yaml` (10 agents)
- `src/quantstack/graphs/supervisor/config/agents.yaml` (3 agents)

For each agent, examine the existing fallback dict in its graph node and the downstream code that consumes the output to determine the required fields and types.

Create an `__init__.py` that exports a registry mapping agent name to its output model:

```python
AGENT_OUTPUT_SCHEMAS: dict[str, type[BaseModel]] = {
    "market_intel": MarketIntelOutput,
    "entry_scan": EntrySignalOutput,
    # ... all 21
}
```

### Step 2: Audit and fix ALL fallback values

**Files to modify:** Node files across all three graphs where `parse_json_response()` is called with a fallback.

Classify every fallback as fail-SAFE or fail-OPEN. The critical fixes:

| Agent | Current Fallback | Problem | Correct Fallback |
|-------|-----------------|---------|-----------------|
| `safety_check` | `{"halted": False}` | Parse failure bypasses safety | `{"halted": True, "reason": "parse_failure"}` |
| `risk_assessment` | (audit needed) | May approve on failure | Must reject on failure |
| `entry_scan` | `[]` or `{"signals": []}` | No entries on failure | Already safe -- no change needed |
| `position_review` | (audit needed) | May trigger exits on failure | No exits triggered (hold positions) |

Every other agent's fallback must be reviewed and documented. The principle: on parse failure, the system should do nothing or halt -- never take an affirmative action (approve, enter, exit) based on a failed parse.

### Step 3: Enhance parse_json_response() to parse_and_validate()

**File:** `src/quantstack/graphs/agent_executor.py`

Rename `parse_json_response()` to `parse_and_validate()`. The new signature:

```python
def parse_and_validate(
    raw_output: str,
    output_schema: type[BaseModel] | None = None,
    agent_name: str = "",
    cycle_id: str = "",
    graph_name: str = "",
) -> tuple[dict, bool]:
    """Parse LLM output as JSON and validate against schema.
    
    Returns (parsed_dict, was_retried) tuple.
    """
```

The logic flow:

1. Attempt JSON parse (preserve existing extraction logic: direct parse, then substring extraction).
2. If JSON parsed and `output_schema` provided: validate via `output_schema.model_validate(parsed_json)`. If valid, return `(parsed_dict, False)`.
3. If validation fails OR JSON parse fails: this is where retry happens (see Step 4).
4. If retry also fails: log at WARNING level, write to dead letter queue (see Step 5), return `(fail_safe_fallback, False)`.

### Step 4: Implement retry mechanism

**File:** `src/quantstack/graphs/agent_executor.py`

The agent executor loop (lines 181-350) already runs a round-based conversation. On parse failure, the retry appends a correction message to the conversation:

```
"Your response was not valid JSON matching the expected schema. 
Please respond with valid JSON matching this schema: {schema_json}"
```

Where `schema_json` is `output_schema.model_json_schema()` serialized as a string.

Run exactly one more LLM round. If this retry also fails validation, proceed to dead letter queue and fallback.

The `execute_agent_node()` function needs an additional parameter:

```python
def execute_agent_node(
    ...,
    output_schema: type[BaseModel] | None = None,
) -> dict:
```

Each graph node passes its expected schema when calling the executor. Nodes that don't yet have schemas continue to work without validation (graceful rollout).

Retried outputs must be flagged in the audit trail. Add a `"_retried": True` key to the returned dict (or use a separate metadata channel if `_retried` pollutes the output). This allows downstream consumers and human reviewers to identify outputs that came from the retry path, which may warrant extra scrutiny.

### Step 5: Implement dead letter queue

**Database table:** `agent_dead_letters`

```sql
CREATE TABLE agent_dead_letters (
    id SERIAL PRIMARY KEY,
    agent_name TEXT NOT NULL,
    cycle_id TEXT NOT NULL,
    graph_name TEXT NOT NULL,
    raw_output TEXT NOT NULL,
    parse_error TEXT NOT NULL,
    retry_attempted BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_dlq_agent_time ON agent_dead_letters (agent_name, created_at);
```

The corresponding Pydantic model for application code:

```python
class AgentDeadLetter(BaseModel):
    agent_name: str
    cycle_id: str
    graph_name: str
    raw_output: str
    parse_error: str
    retry_attempted: bool
    created_at: datetime
```

When `parse_and_validate()` exhausts all attempts (initial + one retry), it writes a row to this table with the raw output and the error message. Use the standard `db_conn()` context manager for the write.

### Step 6: DLQ monitoring in supervisor

**File:** `src/quantstack/graphs/supervisor/nodes.py` (or the appropriate supervisor health check node)

Add a health check query that runs every supervisor cycle:

```sql
SELECT agent_name, 
       COUNT(*) as dlq_count,
       COUNT(*) * 100.0 / NULLIF(total_cycles, 0) as dlq_rate
FROM agent_dead_letters
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY agent_name
HAVING COUNT(*) * 100.0 / NULLIF(total_cycles, 0) > 10;
```

The `total_cycles` value comes from the `graph_checkpoints` table (count of completed cycles per graph in the last 24 hours). If any agent exceeds a 10% DLQ rate, the supervisor should log a warning and optionally publish an event indicating a prompt quality issue for that agent. This is a diagnostic signal, not an automatic remediation -- it flags agents whose prompts need human investigation.

## Key Files

| Action | File |
|--------|------|
| New | `src/quantstack/graphs/schemas/` directory with one model per agent + registry |
| Modify | `src/quantstack/graphs/agent_executor.py` -- rename and enhance parse function, add retry, add output_schema param |
| Modify | `src/quantstack/graphs/trading/nodes.py` -- fix safety_check fallback, pass output schemas |
| Modify | `src/quantstack/graphs/research/nodes.py` -- pass output schemas to executor |
| Modify | `src/quantstack/graphs/supervisor/nodes.py` -- pass output schemas, add DLQ health check |
| New | Migration or init script for `agent_dead_letters` table |
| New | Tests in `tests/unit/test_output_schema_validation.py` and `tests/integration/test_dlq.py` |

## Key Invariants

- `safety_check` MUST fail CLOSED: parse failure produces `{"halted": True}`, never `{"halted": False}`.
- Every agent has exactly one Pydantic output model registered in `AGENT_OUTPUT_SCHEMAS`.
- Retry happens at most once per parse failure. No infinite retry loops.
- Every failed parse (after retry exhaustion) produces a DLQ entry. No silent swallowing.
- DLQ monitoring alerts when any agent exceeds 10% failure rate over 24 hours.
- Retried outputs are explicitly flagged so downstream consumers can distinguish them from first-pass outputs.
