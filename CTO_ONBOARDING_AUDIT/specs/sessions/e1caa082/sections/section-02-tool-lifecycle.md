# Section 02: Tool Lifecycle (AR-8)

## Problem

92 of 122 tools in `TOOL_REGISTRY` are stubs that return errors when invoked. Agents waste LLM round-trips discovering tools exist, calling them, getting errors, and retrying. The deferred loading system (BM25 search in `search_deferred_tools`) surfaces stubs alongside working tools because it matches on descriptions, not health status. There is no mechanism to track which tools are healthy, which are broken, or which planned tools agents actually want.

## Dependencies

- **section-01-db-migrations** must be completed first. This section requires two new tables (`tool_health`, `tool_demand_signals`) that are created in the DB migrations section.
- **section-05-event-bus-extensions** must be completed first. This section publishes `TOOL_ADDED` and `TOOL_DISABLED` events that require the new `EventType` enum values added in that section.

## Blocked By This Section

- **section-07-overnight-autoresearch** depends on the registry split (reads only `ACTIVE_TOOLS`).
- **section-14-autoresclaw-upgrades** depends on the `tool_implement` task type and `tool_manifest.yaml` introduced here.

---

## Tests First

All tests go in `tests/unit/test_tool_lifecycle.py`. Testing framework is pytest with existing fixtures in `tests/unit/conftest.py`.

```python
"""tests/unit/test_tool_lifecycle.py"""

# --- Registry Split ---

# Test: ACTIVE_TOOLS contains only tools with status="active" in manifest
# Load tool_manifest.yaml, filter for status="active", verify ACTIVE_TOOLS keys match exactly.

# Test: PLANNED_TOOLS contains only tools with status="planned" in manifest
# Same approach, filter for status="planned".

# Test: bind_tools_to_llm resolves only from ACTIVE_TOOLS (never PLANNED_TOOLS)
# Call get_tools_for_agent with a tool name that exists in PLANNED_TOOLS but not ACTIVE_TOOLS.
# Expect KeyError.

# Test: search_deferred_tools excludes PLANNED_TOOLS from results
# Add a planned tool with a matching description. Call search_deferred_tools.
# Verify planned tool does NOT appear in results.

# --- Health Monitoring ---

# Test: tool_health_check auto-disables tool with success_rate < 50% over 7 days
# Insert tool_health row with 40% success rate over 7 days. Run health check.
# Verify tool moved from ACTIVE_TOOLS to DEGRADED_TOOLS.

# Test: tool_health_check does NOT disable tool with success_rate = 51%
# Insert tool_health row with 51% success rate. Run health check.
# Verify tool remains in ACTIVE_TOOLS.

# Test: tool_health_check moves disabled tool to DEGRADED_TOOLS
# After auto-disable, verify the tool is accessible in DEGRADED_TOOLS dict,
# not in ACTIVE_TOOLS, and still in TOOL_REGISTRY (backward compat union).

# --- Health Tracking Middleware ---

# Test: health tracking middleware increments counters on success
# Wrap a mock tool with the middleware. Call it successfully.
# Verify invocation_count and success_count both incremented by 1.

# Test: health tracking middleware increments failure_count and records last_error on exception
# Wrap a mock tool that raises. Call it. Verify failure_count incremented
# and last_error contains the exception message.

# --- Demand Signal Tracking ---

# Test: demand_signal_tracker logs search query when planned tool matches
# Call search_deferred_tools with a query that matches a planned tool's description.
# Verify a row was inserted into tool_demand_signals with the query, agent, and tool name.

# Test: demand_signal_weekly_aggregation ranks by frequency correctly
# Insert 5 demand signals for tool A, 2 for tool B, 8 for tool C.
# Run weekly aggregation. Verify ranking: C > A > B.

# --- Tool Synthesis Pipeline ---

# Test: tool_implement task type generates valid tool from planned definition
# Provide a planned tool definition (name, description, expected I/O).
# Verify the task type is accepted by AutoResearchClaw and produces
# a file in tools/langchain/ or tools/functions/.

# --- Event Publishing ---

# Test: TOOL_ADDED event published when tool moves from PLANNED to ACTIVE
# Simulate a tool passing synthesis validation. Verify a TOOL_ADDED event
# is published to the event bus with the tool_name in the payload.

# Test: TOOL_DISABLED event published when tool auto-disabled
# Trigger a health-check auto-disable. Verify a TOOL_DISABLED event
# is published with tool_name, reason, and success_rate in the payload.
```

Integration tests go in `tests/integration/test_tool_lifecycle.py`:

```python
"""tests/integration/test_tool_lifecycle.py"""

# Test: agent receives updated tool set after TOOL_ADDED event (poll + rebind cycle)
# Publish a TOOL_ADDED event. Simulate a graph cycle start (context_load).
# Verify the agent's tool binding includes the newly added tool.

# Test: full lifecycle: planned tool -> demand signal -> synthesis -> active -> agent sees it
# 1. Register a planned tool in tool_manifest.yaml.
# 2. Search for it (generates demand signal).
# 3. Run tool_implement task (synthesis).
# 4. Verify tool moved to ACTIVE_TOOLS.
# 5. Verify TOOL_ADDED event published.
# 6. Verify agent can now bind and invoke the tool.
```

---

## Implementation

### 1. Create `tool_manifest.yaml`

**File:** `src/quantstack/tools/tool_manifest.yaml` (NEW)

A YAML file that classifies every tool by operational status. This is the single source of truth for tool classification, chosen over a decorator-based approach because it is easier to audit and modify without touching Python code.

Structure:

```yaml
# tool_manifest.yaml — tool status classification
# status values: active | planned | degraded | disabled
# test_fixture: optional invocation spec for functional validation (used by AutoResearchClaw)

tools:
  signal_brief:
    status: active
  multi_signal_brief:
    status: active
  fetch_market_data:
    status: active
  # ... all currently working tools listed as active ...

  # Example planned tool (stub that doesn't work yet):
  some_future_tool:
    status: planned
    description: "Short description of what this tool would do"
    expected_input:
      symbol: str
      timeframe: str
    expected_output:
      signal_strength: float
      direction: str
    test_fixture:
      input: { "symbol": "AAPL", "timeframe": "1d" }
      expected_output_keys: ["signal_strength", "direction"]
```

To populate the initial manifest, iterate through the current `TOOL_REGISTRY` and classify each tool: if it can be invoked without error on a simple test input, it is `active`; otherwise it is `planned`. This is a one-time audit.

### 2. Modify `registry.py` — Registry Split

**File:** `src/quantstack/tools/registry.py` (MODIFY)

The existing `TOOL_REGISTRY` dict remains as a computed union for backward compatibility, but all agent-facing code reads from `ACTIVE_TOOLS` only.

Key changes:

- At module load, read `tool_manifest.yaml` from the same directory as `registry.py`.
- Build three dicts: `ACTIVE_TOOLS`, `PLANNED_TOOLS`, `DEGRADED_TOOLS`.
- Classification: for each tool in `TOOL_REGISTRY`, look up its status in the manifest. Tools not in the manifest default to `active` (backward compatible).
- `TOOL_REGISTRY` becomes `{**ACTIVE_TOOLS, **PLANNED_TOOLS, **DEGRADED_TOOLS}` for any code that still references it.

Functions to modify:

- **`get_tools_for_agent`**: Change to resolve from `ACTIVE_TOOLS` only. Raise `KeyError` if a tool name is in `PLANNED_TOOLS` or `DEGRADED_TOOLS` (with a message indicating why).
- **`get_tools_for_agent_with_search`**: Same — resolve from `ACTIVE_TOOLS` only.
- **`search_deferred_tools`**: Filter the `deferred_names` set to exclude any name in `PLANNED_TOOLS` before scoring. This prevents agents from discovering stubs via search.

New functions to add:

```python
def load_tool_manifest() -> dict[str, dict]:
    """Load tool_manifest.yaml and return {tool_name: {status, ...}}."""
    # Read from Path(__file__).parent / "tool_manifest.yaml"
    # Return parsed YAML dict

def classify_tools() -> None:
    """Partition TOOL_REGISTRY into ACTIVE_TOOLS, PLANNED_TOOLS, DEGRADED_TOOLS
    based on the manifest. Called once at module load and again on TOOL_ADDED/TOOL_DISABLED events."""

def move_tool(tool_name: str, from_status: str, to_status: str) -> None:
    """Move a tool between registries (e.g., planned -> active after synthesis).
    Updates the in-memory dicts. Does NOT update the YAML file (that is a separate commit)."""
```

Module-level variables to add:

```python
ACTIVE_TOOLS: dict[str, BaseTool] = {}
PLANNED_TOOLS: dict[str, BaseTool] = {}
DEGRADED_TOOLS: dict[str, BaseTool] = {}
```

### 3. Add Health Tracking Middleware

**File:** `src/quantstack/tools/_helpers.py` (MODIFY)

Add a middleware wrapper that instruments every tool invocation. This goes between the tool's `@tool` decorator and the actual function body, applied at registration time or via a wrapper in `_helpers.py`.

```python
def track_tool_health(tool_name: str, success: bool, latency_ms: float, error: str | None = None) -> None:
    """Record a tool invocation result to the tool_health table.

    Uses a fire-and-forget pattern — health tracking failures must never
    block tool execution. Catches and logs all exceptions internally.

    Args:
        tool_name: The registered tool name.
        success: Whether the invocation completed without raising.
        latency_ms: Wall-clock execution time in milliseconds.
        error: Exception message if success is False.
    """
    # INSERT INTO tool_health ... ON CONFLICT (tool_name) DO UPDATE
    # SET invocation_count = invocation_count + 1,
    #     success_count = success_count + (1 if success else 0),
    #     failure_count = failure_count + (0 if success else 1),
    #     avg_latency_ms = <running average>,
    #     last_invoked = NOW(),
    #     last_error = error (if not None)
```

The middleware wraps tool execution in `try/finally` to ensure latency and outcome are always recorded, even on exception. The tool's original exception is re-raised after recording.

Design constraint: health tracking uses its own DB connection (not the graph's transaction) so a health-tracking failure never rolls back a tool's actual work.

### 4. Create Health Monitor

**File:** `src/quantstack/tools/health_monitor.py` (NEW)

A module that runs as part of the supervisor graph's daily health check. Responsibilities:

- Query `tool_health` table for trailing 7-day success rates per tool.
- For any tool with `success_rate < 0.50` (i.e., more failures than successes over 7 days): call `move_tool(tool_name, "active", "degraded")` and publish a `TOOL_DISABLED` event to the event bus.
- The 50% threshold is aggressive enough to catch broken tools within a week but forgiving enough to tolerate intermittent API failures (e.g., a tool that fails 2/10 times due to upstream rate limits stays active at 80%).
- Tools in `DEGRADED_TOOLS` can be re-enabled manually or by AutoResearchClaw after a fix.

Key function:

```python
def run_daily_health_check(event_bus: EventBus) -> list[str]:
    """Check all active tools' 7-day health and auto-disable degraded ones.

    Returns list of tool names that were disabled.
    """
```

### 5. Demand Signal Tracking

**File:** `src/quantstack/tools/registry.py` (MODIFY — within `search_deferred_tools`)

When `search_deferred_tools` finds a match against a tool in `PLANNED_TOOLS`, log the demand signal to the `tool_demand_signals` table:

```sql
INSERT INTO tool_demand_signals (signal_id, tool_name, search_query, requesting_agent, created_at)
VALUES (gen_random_uuid(), ?, ?, ?, NOW())
```

The `requesting_agent` comes from the graph state's current agent ID (passed as a parameter to `search_deferred_tools`).

New function for weekly aggregation:

```python
def aggregate_demand_signals() -> list[dict]:
    """Aggregate tool_demand_signals over the trailing 7 days.

    Returns a ranked list of planned tools by demand frequency:
    [{"tool_name": "...", "demand_count": N, "unique_agents": M}, ...]
    sorted by demand_count descending.

    The top 3 become tool_implement tasks in research_queue.
    """
```

This aggregation runs weekly as part of the supervisor graph or as a scheduled job.

### 6. Tool Synthesis Pipeline (AutoResearchClaw Extension)

**File:** `scripts/autoresclaw_runner.py` (MODIFY)

Add a new task type `tool_implement` alongside the existing `bug_fix`, `ml_arch_search`, and `strategy_hypothesis` types.

Input for a `tool_implement` task:
- Tool name from `PLANNED_TOOLS`
- Description, expected input schema, expected output schema (from `tool_manifest.yaml`)

Processing steps:
1. Generate implementation using Opus (existing AutoResearchClaw LLM tier).
2. Write the file to `src/quantstack/tools/langchain/<tool_name>.py` or `src/quantstack/tools/functions/<tool_name>.py`.
3. Validate: `py_compile` + import + invoke with test fixture from manifest.
4. If validation passes: update `tool_manifest.yaml` status to `active`, call `move_tool` to move from `PLANNED_TOOLS` to `ACTIVE_TOOLS`, publish `TOOL_ADDED` event.
5. If validation fails: log failure, leave tool in `planned` status.

The test fixture validation is a key improvement over the current AutoResearchClaw validation (which only runs `py_compile` + import). The fixture provides known inputs and expected output keys, enabling a functional smoke test.

### 7. Capability Announcement (Event Bus Integration)

This section requires the `TOOL_ADDED` and `TOOL_DISABLED` event types from **section-05-event-bus-extensions**. The integration points are:

- **Publishing**: `health_monitor.py` publishes `TOOL_DISABLED` when auto-disabling. The synthesis pipeline publishes `TOOL_ADDED` when a planned tool passes validation.
- **Consuming**: Research and trading graphs poll for these events at cycle start in their `context_load` nodes (existing polling infrastructure). On `TOOL_ADDED`, the graph calls `classify_tools()` to refresh the in-memory registries and rebuilds its tool binding. On `TOOL_DISABLED`, same refresh — the disabled tool disappears from `ACTIVE_TOOLS`.

Event payloads:

```python
# TOOL_ADDED
{"tool_name": "new_tool_name", "source": "synthesis"}  # or "manual"

# TOOL_DISABLED
{"tool_name": "broken_tool", "reason": "health_check", "success_rate": 0.42}
```

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `src/quantstack/tools/tool_manifest.yaml` | CREATE | Tool status classification (active/planned/degraded/disabled) |
| `src/quantstack/tools/registry.py` | MODIFY | Split into ACTIVE/PLANNED/DEGRADED, read manifest, demand signal logging |
| `src/quantstack/tools/_helpers.py` | MODIFY | Add health tracking middleware (`track_tool_health`) |
| `src/quantstack/tools/health_monitor.py` | CREATE | Daily health check, auto-disable logic |
| `scripts/autoresclaw_runner.py` | MODIFY | Add `tool_implement` task type |
| `tests/unit/test_tool_lifecycle.py` | CREATE | Unit tests for all components |
| `tests/integration/test_tool_lifecycle.py` | CREATE | Integration tests for full lifecycle |

---

## Key Design Decisions

1. **Manifest over decorator**: `tool_manifest.yaml` is the source of truth for tool status. The alternative (a `status` field in each `@tool` decorator) keeps everything co-located but requires Python changes to reclassify. The manifest approach allows operational changes (disabling a broken tool) without code changes.

2. **Health check frequency**: Daily, not real-time. Real-time health monitoring would add per-invocation latency to every tool call. The 7-day trailing window smooths transient failures. The per-invocation middleware only writes counters (fire-and-forget); the actual health decision happens once per day.

3. **Auto-disable threshold**: 50% success rate over 7 days. This means a tool must fail more often than it succeeds over a full week before being disabled. A tool that fails 2 out of 10 calls (80% success) stays active. A tool that fails 6 out of 10 calls (40% success) gets disabled.

4. **Backward compatibility**: `TOOL_REGISTRY` remains as a union of all three dicts. Any code that references `TOOL_REGISTRY` directly continues to work. Only the agent-facing functions (`get_tools_for_agent`, `get_tools_for_agent_with_search`, `search_deferred_tools`) change to read from `ACTIVE_TOOLS` only. This is the expand-contract pattern — new code uses the split registries, old code keeps working.

5. **Demand signals drive synthesis**: Rather than synthesizing all planned tools at once, the system waits for agents to actually search for a tool before prioritizing its implementation. This ensures development effort goes to tools agents actually need, not tools someone thought might be useful.
