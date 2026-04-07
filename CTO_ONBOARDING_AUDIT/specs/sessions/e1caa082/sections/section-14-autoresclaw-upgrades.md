# Section 14: AutoResearchClaw Upgrades

## Overview

AutoResearchClaw (ARC) is an autonomous task runner (`scripts/autoresclaw_runner.py`, 683 lines) that translates `research_queue` tasks into sandboxed Claude Code invocations. It currently handles 4 task types (`bug_fix`, `ml_arch_search`, `rl_env_design`, `strategy_hypothesis`), runs only on Sundays at 20:00 ET, restarts affected services via `tmux send-keys`, and validates fixes with `py_compile` + import checks only.

Phase 10 requires 4 upgrades to close gaps that block the overnight autoresearch pipeline, tool lifecycle, and error-driven research systems:

1. **Nightly schedule** (was Sunday-only)
2. **Two new proactive task types** (`tool_implement`, `gap_detection`)
3. **Docker Compose restarts** (replace fragile tmux mechanism)
4. **Functional validation** with test fixtures (beyond py_compile)

## Dependencies

- **section-02-tool-lifecycle**: Provides `tool_manifest.yaml` with planned tool definitions and `tool_demand_signals` table that feeds `tool_implement` tasks. Also defines the `PLANNED_TOOLS` registry that ARC reads from.
- **section-05-event-bus-extensions**: Provides `TOOL_ADDED` event type that ARC publishes after successful tool implementation.

## Current State (What Exists)

### `scripts/autoresclaw_runner.py`

- **Prompt builders**: `_PROMPT_BUILDERS` dict maps task_type strings to functions that produce markdown prompts for Claude Code. Each builder receives a `ctx` dict (from `research_queue.context_json`) and returns the full prompt.
- **Core loop**: `process_tasks()` iterates tasks sequentially, calling `run_task()` which invokes the `researchclaw` CLI with `--auto-approve --agent claude-code`.
- **Bug fix post-processing**: `_apply_bug_fix()` reads the generated `fix_summary.md`, checks for low confidence and protected file violations, runs `py_compile` on changed files, commits valid patches, and restarts affected loops.
- **Protected files**: `_PROTECTED_FILES = frozenset(["risk_gate.py", "kill_switch.py", "db.py"])` -- never modified by ARC.
- **Restart mechanism**: `_restart_loops_after_fix()` uses `tmux send-keys` to send C-c and restart commands to named tmux windows (`quantstack-loops:trading`, `quantstack-loops:research`).

### `scripts/scheduler.py`

- ARC is scheduled via `run_autoresclaw_weekly()` which runs `python scripts/autoresclaw_runner.py --limit 3`.
- Current schedule entry: `{"trigger": {"hour": 20, "minute": 0, "day_of_week": "sun"}, "func": run_autoresclaw_weekly, "label": "autoresclaw_weekly_sun20:00"}`.

### `src/quantstack/tools/tool_manifest.yaml`

Does not exist yet. Created by section-02-tool-lifecycle. Each entry will have: tool name, status (`active`/`planned`/`degraded`/`disabled`), description, expected input/output schema, and a `test_fixture` field.

---

## Tests First

All tests go in `tests/unit/test_autoresclaw_upgrades.py`.

```python
"""Tests for AutoResearchClaw upgrades — Phase 10 Section 14.

Validates nightly scheduling, new task types, Docker Compose restarts,
and functional validation with test fixtures.
"""

import pytest


# --- Task type acceptance ---

def test_tool_implement_task_type_accepted_and_processed():
    """tool_implement task type has a prompt builder in _PROMPT_BUILDERS
    and produces a valid prompt string from a context dict containing
    tool_name, description, expected_input, and expected_output fields."""


def test_gap_detection_task_type_accepted_and_processed():
    """gap_detection task type has a prompt builder in _PROMPT_BUILDERS
    and produces a valid prompt from a context dict containing
    failure_mode, affected_strategies, example_losses, and
    suggested_research_direction fields."""


# --- Functional validation ---

def test_functional_validation_invokes_tool_test_fixture():
    """After a tool_implement or bug_fix patch, validation runs the tool's
    test fixture (defined in tool_manifest.yaml) rather than only py_compile.
    The fixture is a simple invocation with known inputs that must return
    a result matching the expected output shape."""


def test_functional_validation_reverts_patch_on_test_fixture_failure():
    """When the test fixture invocation fails (raises exception or returns
    unexpected shape), the patch is reverted via git checkout, the task is
    marked 'failed' with reason 'test_fixture_failed', and a note is written
    to session_handoffs.md."""


# --- Docker Compose restarts ---

def test_docker_compose_restart_called_instead_of_tmux():
    """_restart_loops_after_fix calls 'docker compose restart <service>'
    for affected services instead of 'tmux send-keys'. The subprocess.run
    call targets 'docker compose restart trading-graph' and/or
    'docker compose restart research-graph' based on which source files
    were changed."""


# --- Nightly schedule ---

def test_nightly_schedule_triggers():
    """The scheduler JOBS list contains an autoresclaw entry that runs
    nightly (every day at 20:00 ET), not just Sunday. The trigger dict
    should NOT have day_of_week='sun'."""
```

---

## Implementation Details

### 1. New Prompt Builders (`scripts/autoresclaw_runner.py`)

Add two new prompt builder functions and register them in `_PROMPT_BUILDERS`.

**`_build_prompt_tool_implement(ctx)`**

Context keys (from `tool_demand_signals` aggregation via section-02):
- `tool_name`: Name of the planned tool to implement
- `description`: Tool description from `tool_manifest.yaml`
- `expected_input`: Pydantic model or dict describing input schema
- `expected_output`: Pydantic model or dict describing output schema
- `demand_count`: Number of times agents searched for this tool
- `test_fixture`: Dict with `input` and `expected_output_shape` for validation

The prompt should instruct Claude Code to:
1. Read the planned tool definition from `tool_manifest.yaml`
2. Implement the tool in `src/quantstack/tools/langchain/` (if LLM-facing) or `src/quantstack/tools/functions/` (if deterministic)
3. Follow existing patterns: use `@tool` decorator, Pydantic I/O models in `tools/models.py`, register in `tools/registry.py`
4. Validate: py_compile, import, invoke with the provided test fixture inputs
5. On success: move the tool from `PLANNED_TOOLS` to `ACTIVE_TOOLS` in the manifest, fire `TOOL_ADDED` event

**`_build_prompt_gap_detection(ctx)`**

Context keys (from loss analyzer via section-03):
- `failure_mode`: The classified failure mode (e.g., `regime_mismatch`, `liquidity_trap`)
- `affected_strategies`: List of strategy_ids affected
- `example_losses`: List of example losing trades with metadata
- `suggested_research_direction`: Human-readable suggestion from the loss analyzer
- `cumulative_pnl_impact`: Dollar amount of total losses from this failure mode

The prompt should instruct Claude Code to:
1. Analyze the failure mode and example losses
2. Search literature for mitigation strategies (web search)
3. Generate `research_queue` tasks targeting the gap
4. Each task should be a `strategy_hypothesis` with context linking back to the failure mode

Register both in `_PROMPT_BUILDERS`:

```python
_PROMPT_BUILDERS = {
    "ml_arch_search": _build_prompt_ml_arch_search,
    "rl_env_design": _build_prompt_rl_env_design,
    "bug_fix": _build_prompt_bug_fix,
    "strategy_hypothesis": _build_prompt_strategy_hypothesis,
    "tool_implement": _build_prompt_tool_implement,      # NEW
    "gap_detection": _build_prompt_gap_detection,         # NEW
}
```

### 2. Functional Validation (`scripts/autoresclaw_runner.py`)

Replace the py_compile-only validation in `_apply_bug_fix()` with a two-stage validation:

**Stage 1 (existing):** `py_compile` + import check for all changed `.py` files. This catches syntax errors and import failures.

**Stage 2 (new):** For `tool_implement` and `bug_fix` tasks, look up the tool's `test_fixture` in `tool_manifest.yaml`. If a fixture exists:

1. Parse the fixture: `{"input": {...}, "expected_output_shape": {"type": "dict", "keys": ["result", "confidence"]}}`
2. Import the tool function
3. Invoke it with the fixture input
4. Validate the output matches the expected shape (type check, required keys present)
5. If validation fails: revert the patch (same `_revert_and_note` path), mark task `failed` with reason `test_fixture_failed`

The fixture lookup requires reading `tool_manifest.yaml`. Use a helper function:

```python
def _load_test_fixture(tool_name: str) -> dict | None:
    """Load test fixture for a tool from tool_manifest.yaml.

    Returns None if manifest doesn't exist or tool has no fixture.
    """
```

**Fallback behavior:** If `tool_manifest.yaml` does not exist yet (section-02 not implemented) or the tool has no `test_fixture` entry, fall back to the existing py_compile-only validation. This ensures backward compatibility during incremental rollout.

### 3. Docker Compose Restarts (`scripts/autoresclaw_runner.py`)

Replace `_restart_loops_after_fix()` entirely. The current implementation uses `tmux send-keys` which fails silently when:
- The tmux session doesn't exist (Docker deployment)
- The window name has changed
- The tmux server isn't running

New implementation:

```python
def _restart_loops_after_fix(changed_files: list[str]) -> None:
    """Restart affected Docker Compose services after a code fix.

    Determines which services to restart based on which source directories
    were modified. Uses 'docker compose restart' which sends SIGTERM,
    waits for graceful shutdown, then restarts the container.
    """
```

Service mapping (same heuristic as current, different restart mechanism):
- Files in `signal/`, `execution/`, `data/`, `coordination/` --> `docker compose restart trading-graph`
- Files in `research/`, `ml/`, `models/`, `features/` --> `docker compose restart research-graph`
- If unclear which service is affected --> restart both

The `subprocess.run` call should:
- Use `["docker", "compose", "restart", service_name]` (not `docker-compose` -- the project uses Docker Compose v2)
- Set `cwd=str(WORKDIR)` so it finds `docker-compose.yml`
- Set `timeout=60` (containers should restart within 60 seconds)
- Log success/failure per service
- Catch `FileNotFoundError` (docker not installed) and `TimeoutExpired` gracefully

### 4. Nightly Schedule (`scripts/scheduler.py`)

Change the ARC scheduler entry from Sunday-only to nightly.

**Current entry in `JOBS` list:**
```python
{"trigger": {"hour": 20, "minute": 0, "day_of_week": "sun"}, "func": run_autoresclaw_weekly, "label": "autoresclaw_weekly_sun20:00"},
```

**New entry:**
```python
{"trigger": {"hour": 20, "minute": 0}, "func": run_autoresclaw_nightly, "label": "autoresclaw_nightly_20:00"},
```

Note: removing `day_of_week` makes the trigger fire every day at 20:00 ET.

**Rename the job function** from `run_autoresclaw_weekly` to `run_autoresclaw_nightly` to match the new cadence. Update the docstring and label accordingly. The function body remains the same (it calls `python scripts/autoresclaw_runner.py --limit 3`), but the semantic change is important for operational clarity.

**Separate weekly slot.** The plan states: "AutoResearchClaw's weekly bug_fix/ml_arch_search tasks move to a separate weekly slot." However, this separation is handled by task priority in `research_queue`, not by scheduler changes. The nightly runner processes the top-N tasks regardless of type. Bug fixes and ml_arch_search tasks are queued with appropriate priorities by the supervisor graph and drift detector respectively. No additional scheduler entry is needed for the weekly slot -- the existing priority ordering ensures urgent tasks (bug_fix) are processed first.

**Update `func_map` in `main()`** for `--run-now`:
```python
"autoresclaw_nightly": run_autoresclaw_nightly,
```

Remove the old `autoresclaw_weekly` entry from `func_map`.

**Update the startup banner** in `start_scheduler()` to reflect the nightly schedule:
```
  20:00 Daily       -- AutoResearchClaw task processing (research_queue)
```

### 5. `tool_manifest.yaml` Test Fixture Schema

When section-02 creates `tool_manifest.yaml`, each tool entry should include an optional `test_fixture` field. This section defines the schema ARC expects:

```yaml
tools:
  compute_risk_metrics:
    status: active
    description: "Compute VaR, CVaR, max drawdown for a portfolio"
    test_fixture:
      input:
        returns: [0.01, -0.02, 0.015, -0.005, 0.008]
        confidence_level: 0.95
      expected_output_shape:
        type: dict
        required_keys: ["var", "cvar", "max_drawdown"]

  predict_regime_transition:
    status: planned
    description: "Predict probability of regime change in next 5 days"
    test_fixture:
      input:
        symbol: "SPY"
        lookback_days: 60
      expected_output_shape:
        type: dict
        required_keys: ["transition_probability", "current_regime", "predicted_regime"]
```

The `expected_output_shape` is intentionally simple: check type and required keys. Deep value validation is left to the tool's own unit tests. The fixture is a smoke test that confirms the tool can be imported, invoked, and returns the right structure.

---

## Files to Create/Modify

| File | Action | What Changes |
|------|--------|-------------|
| `scripts/autoresclaw_runner.py` | MODIFY | Add `_build_prompt_tool_implement`, `_build_prompt_gap_detection`, register in `_PROMPT_BUILDERS`. Replace `_restart_loops_after_fix` with Docker Compose version. Add `_load_test_fixture` and `_run_functional_validation` helpers. Update `_apply_bug_fix` to call functional validation. |
| `scripts/scheduler.py` | MODIFY | Rename `run_autoresclaw_weekly` to `run_autoresclaw_nightly`. Change JOBS trigger from `day_of_week="sun"` to no day restriction. Update `func_map`, startup banner, and docstring. |
| `src/quantstack/tools/tool_manifest.yaml` | MODIFY (created by section-02) | Add `test_fixture` field schema to each tool entry. |
| `tests/unit/test_autoresclaw_upgrades.py` | CREATE | 6 unit tests covering task types, validation, restart, and schedule. |

---

## Verification Checklist

After implementation, verify:

1. `python scripts/autoresclaw_runner.py --dry-run` shows both new task types in the prompt builder list
2. Insert a `tool_implement` task into `research_queue` and run `--dry-run --task-id <uuid>` -- confirm the prompt includes the tool definition and test fixture instructions
3. Insert a `gap_detection` task and verify the prompt includes failure mode context
4. `python scripts/scheduler.py --dry-run` shows `autoresclaw_nightly_20:00` with no `day_of_week` restriction
5. `docker compose restart trading-graph` works from the project root (manual test)
6. All 6 tests in `test_autoresclaw_upgrades.py` pass: `uv run pytest tests/unit/test_autoresclaw_upgrades.py -v`
