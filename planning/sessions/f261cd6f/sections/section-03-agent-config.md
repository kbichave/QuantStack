# Section 3: Agent Configuration System

## Overview

This section builds a YAML-based agent configuration system that replaces CrewAI's implicit agent setup with explicit, validated, hot-reloadable config. The system has three parts: (1) a new YAML format for agent profiles, (2) an `AgentConfig` dataclass with validation and a loader, and (3) a `ConfigWatcher` class that supports hot-reload via file-watching (dev) and SIGHUP (prod).

The configuration system is consumed by all three graph builders (supervisor, research, trading) in sections 06-08. Each graph builder reads agent configs from the `ConfigWatcher` at cycle start and constructs LLM agents accordingly. Because graphs are rebuilt every cycle (they are cheap to build), config changes take effect at the next cycle boundary without any mid-execution disruption.

## Dependencies

- **Section 01 (Scaffolding)**: The `src/quantstack/graphs/` directory structure must exist. The `watchdog>=4.0.0` dependency must be in `pyproject.toml`.
- **Section 05 (Tool Layer)**: The `TOOL_REGISTRY` dictionary must exist for cross-validation of tool references in agent configs. However, the config system can be built and tested independently -- tool cross-validation is a runtime check that can be deferred.

## File Locations

| File | Purpose |
|------|---------|
| `src/quantstack/graphs/config.py` | `AgentConfig` dataclass, `load_agent_configs()` loader |
| `src/quantstack/graphs/config_watcher.py` | `ConfigWatcher` class with hot-reload |
| `src/quantstack/graphs/research/config/agents.yaml` | Research agent profiles (new format) |
| `src/quantstack/graphs/trading/config/agents.yaml` | Trading agent profiles (new format) |
| `src/quantstack/graphs/supervisor/config/agents.yaml` | Supervisor agent profiles (new format) |
| `tests/unit/test_agent_config.py` | All unit tests for this section |

## Tests (Write First)

All tests go in `tests/unit/test_agent_config.py`. Write these before implementation.

```python
"""Tests for agent configuration system."""
import os
import signal
import tempfile
import threading
import time
from pathlib import Path

import pytest
import yaml


# --- AgentConfig dataclass and loader tests ---

def test_agent_config_loads_from_valid_yaml():
    """AgentConfig loads from valid YAML with all fields populated."""
    # Write a valid agents.yaml to a temp file, call load_agent_configs(),
    # assert all fields on the returned AgentConfig match YAML values.

def test_agent_config_rejects_missing_required_fields():
    """AgentConfig rejects YAML missing required fields (role, goal, llm_tier)."""
    # Write YAML with role omitted. Expect ValueError from load_agent_configs().

def test_agent_config_rejects_invalid_llm_tier():
    """AgentConfig rejects llm_tier values not in {"heavy", "medium", "light"}."""
    # Write YAML with llm_tier: "turbo". Expect ValueError.

def test_config_loader_returns_dict_of_agent_configs():
    """load_agent_configs() returns dict[str, AgentConfig] for multi-agent YAML."""
    # Write YAML with 3 agents. Assert return type and length.

def test_config_loader_raises_on_duplicate_agent_names():
    """load_agent_configs() raises on duplicate agent names within same file."""
    # YAML spec merges duplicate keys silently, so the loader must detect
    # this by parsing raw YAML or using a custom loader.

def test_tool_references_cross_validate_against_registry():
    """Tool names in AgentConfig.tools must exist in TOOL_REGISTRY."""
    # Pass a mock TOOL_REGISTRY. Assert ValueError for unknown tool name.


# --- ConfigWatcher tests ---

def test_config_watcher_get_config_returns_current():
    """ConfigWatcher.get_config(name) returns the current AgentConfig."""
    # Initialize ConfigWatcher with a valid YAML path.
    # Assert get_config("quant_researcher") returns an AgentConfig.

def test_config_watcher_detects_file_change_and_reloads():
    """ConfigWatcher detects file change and reloads (dev mode)."""
    # Write initial YAML, start ConfigWatcher in dev mode.
    # Modify the YAML file (change a role string).
    # Wait briefly for watchdog event propagation.
    # Assert get_config() returns the updated role.

def test_config_watcher_handles_sighup_and_reloads():
    """ConfigWatcher handles SIGHUP and reloads (prod mode)."""
    # Write initial YAML, start ConfigWatcher in prod mode.
    # Modify the YAML file.
    # Send SIGHUP to the current process.
    # Assert get_config() returns the updated config.

def test_config_watcher_reload_is_atomic():
    """Concurrent reads during reload never see partial state."""
    # Start ConfigWatcher. Spawn reader threads that call get_config()
    # in a tight loop. Trigger reload from another thread.
    # Assert no reader ever gets None or a half-populated config.

def test_config_watcher_reload_at_cycle_boundary_only():
    """Reload flag is set on file change but config swap happens only
    when the caller explicitly calls apply_pending_reload()."""
    # Modify YAML. Assert get_config() still returns OLD config.
    # Call apply_pending_reload(). Assert get_config() returns NEW config.
```

## YAML Format

The new format strips CrewAI-specific fields (`memory`, `verbose`, `allow_delegation`, `llm` as a template string) and replaces them with fields the graph builder needs directly.

### Example: Research agents.yaml

```yaml
quant_researcher:
  role: "Senior Quantitative Researcher"
  goal: "Discover and validate alpha-generating strategies"
  backstory: |
    You manage a research program with 3-5 active investigations...
    (full backstory text from existing crews/research/config/agents.yaml)
  llm_tier: heavy
  max_iterations: 20
  timeout_seconds: 600
  tools:
    - signal_brief
    - multi_signal_brief
    - fetch_market_data
    - compute_features
    - search_knowledge_base

ml_scientist:
  role: "Machine Learning Scientist"
  goal: "Design training experiments, validate features, manage model lifecycle"
  backstory: |
    (full backstory from existing file)
  llm_tier: heavy
  max_iterations: 20
  timeout_seconds: 600
  tools:
    - fetch_market_data
    - compute_features
    - train_model
    - search_knowledge_base
```

### Field mapping from CrewAI format

| Old field (CrewAI) | New field | Notes |
|---------------------|-----------|-------|
| `llm: "{heavy_model}"` | `llm_tier: heavy` | Graph builder calls `get_chat_model(tier)` |
| `max_iter: 20` | `max_iterations: 20` | Consistent naming |
| `max_execution_time: 600` | `timeout_seconds: 600` | Same semantics |
| `memory: true` | (removed) | State graph handles memory explicitly |
| `verbose: true` | (removed) | Observability via LangFuse callbacks |
| `allow_delegation: false` | (removed) | Graph topology handles delegation |
| (not present) | `tools: [...]` | Tool names referencing TOOL_REGISTRY keys |

### Tier mapping

The `llm_tier` field accepts exactly three values:

- `heavy` -- used for reasoning-intensive agents (quant_researcher, trade_debater, risk_analyst, fund_manager, options_analyst, strategy_rd)
- `medium` -- used for structured-output agents (daily_planner, position_monitor, trade_reflector, earnings_analyst, market_intel, strategy_promoter, executor)
- `light` -- used for simple/fast agents (health_monitor, self_healer, community_intel)

The graph builder resolves these tiers to actual `BaseChatModel` instances via `get_chat_model(tier)` from section 02.

## AgentConfig Dataclass

In `src/quantstack/graphs/config.py`:

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_VALID_TIERS = frozenset({"heavy", "medium", "light"})


@dataclass(frozen=True)
class AgentConfig:
    """Immutable agent profile loaded from YAML."""

    name: str
    role: str
    goal: str
    backstory: str
    llm_tier: str                   # "heavy", "medium", "light"
    max_iterations: int = 20
    timeout_seconds: int = 600
    tools: tuple[str, ...] = ()     # tool registry keys (tuple for immutability)

    def __post_init__(self) -> None:
        if self.llm_tier not in _VALID_TIERS:
            raise ValueError(
                f"Invalid llm_tier '{self.llm_tier}' for agent '{self.name}'. "
                f"Must be one of: {sorted(_VALID_TIERS)}"
            )
        if not self.role:
            raise ValueError(f"Agent '{self.name}' missing required field: role")
        if not self.goal:
            raise ValueError(f"Agent '{self.name}' missing required field: goal")
```

The dataclass is frozen (immutable) so it can be safely shared across threads and cycles. The `tools` field is a tuple rather than a list for the same reason.

## Config Loader

Also in `src/quantstack/graphs/config.py`:

```python
def load_agent_configs(yaml_path: Path) -> dict[str, AgentConfig]:
    """Load and validate agent configs from a YAML file.

    Returns a mapping of agent_name -> AgentConfig.
    Raises ValueError on validation failures (missing fields, invalid tiers,
    duplicate agent names).
    """
```

Implementation notes:

- Parse the YAML file. Each top-level key is an agent name.
- For each entry, construct an `AgentConfig` with `name` set to the key. Convert the `tools` list to a tuple.
- Detect duplicate keys: YAML silently merges duplicates. Either use a custom YAML loader that raises on duplicates, or parse raw text to detect them.
- Return a `dict[str, AgentConfig]`.

Optional cross-validation against a tool registry:

```python
def validate_tool_references(
    configs: dict[str, AgentConfig],
    tool_registry: dict[str, Any],
) -> None:
    """Raise ValueError if any agent references a tool not in the registry."""
```

This is a separate function (not in the loader) because the tool registry may not be available at import time. Graph builders call this after both configs and tools are loaded.

## ConfigWatcher

In `src/quantstack/graphs/config_watcher.py`:

```python
class ConfigWatcher:
    """Loads agent configs and supports hot-reload.

    Two reload mechanisms:
    - Dev mode (file-watch): uses watchdog to monitor YAML files for changes.
    - Prod mode (SIGHUP): registers a signal handler that sets a reload flag.

    In both modes, the actual config swap happens only when the caller invokes
    apply_pending_reload(). This ensures reload occurs at cycle boundaries,
    never mid-graph-execution.
    """

    def __init__(self, yaml_path: Path, *, watch: bool = False) -> None:
        """Load initial configs. If watch=True, start file watcher (dev mode)."""

    def get_config(self, agent_name: str) -> AgentConfig:
        """Return the current config for the named agent. Thread-safe."""

    def get_all_configs(self) -> dict[str, AgentConfig]:
        """Return all current configs. Thread-safe."""

    def apply_pending_reload(self) -> bool:
        """If a reload is pending, atomically swap configs.

        Returns True if configs were reloaded, False if no change pending.
        Called by the runner at the start of each cycle.
        """

    def register_sighup_handler(self) -> None:
        """Register SIGHUP handler for prod-mode reload."""

    def stop(self) -> None:
        """Stop file watcher if running. Clean up resources."""
```

### Concurrency design

The `ConfigWatcher` holds two attributes:

- `_configs: dict[str, AgentConfig]` -- the current active configs, read by graph builders.
- `_pending_configs: dict[str, AgentConfig] | None` -- staged reload, set by watcher thread or signal handler.

Both are protected by a `threading.Lock`. The `get_config()` and `get_all_configs()` methods acquire the lock to read `_configs`. The file watcher or SIGHUP handler acquires the lock to write `_pending_configs`. The `apply_pending_reload()` method acquires the lock and atomically swaps `_configs = _pending_configs; _pending_configs = None`.

This two-phase design ensures:
1. Readers never see partial state (atomic dict reference swap).
2. Reload only takes effect when the runner explicitly requests it (cycle boundary).
3. File watcher errors (bad YAML) are caught at parse time and logged without corrupting active configs.

### File watcher (dev mode)

Uses `watchdog.observers.Observer` to monitor the YAML file's parent directory for `FileModifiedEvent`. On event, parse the file and set `_pending_configs` if valid. Log and skip if YAML is invalid.

### SIGHUP handler (prod mode)

`register_sighup_handler()` calls `signal.signal(signal.SIGHUP, handler)` where the handler re-reads the YAML file and sets `_pending_configs`. Signal handlers run on the main thread in Python, so the lock acquisition is safe.

## YAML Files to Create

Create three new YAML files by adapting the existing CrewAI agent configs. The backstory text should be preserved verbatim -- it contains domain knowledge and decision frameworks that agents need.

### `src/quantstack/graphs/research/config/agents.yaml`

Agents to include (from `crews/research/config/agents.yaml`):
- `quant_researcher` (llm_tier: heavy)
- `ml_scientist` (llm_tier: heavy)
- `strategy_rd` (llm_tier: heavy)
- `community_intel` (llm_tier: light)

### `src/quantstack/graphs/trading/config/agents.yaml`

Agents to include (from `crews/trading/config/agents.yaml`):
- `daily_planner` (llm_tier: medium)
- `position_monitor` (llm_tier: medium)
- `trade_debater` (llm_tier: heavy)
- `risk_analyst` (llm_tier: heavy)
- `fund_manager` (llm_tier: heavy)
- `options_analyst` (llm_tier: heavy)
- `earnings_analyst` (llm_tier: medium)
- `market_intel` (llm_tier: medium)
- `trade_reflector` (llm_tier: medium)
- `executor` (llm_tier: medium)

### `src/quantstack/graphs/supervisor/config/agents.yaml`

Agents to include (from `crews/supervisor/config/agents.yaml`):
- `health_monitor` (llm_tier: light)
- `self_healer` (llm_tier: light)
- `strategy_promoter` (llm_tier: medium)

For each agent, the conversion is mechanical: drop `memory`, `verbose`, `allow_delegation`, and the `llm` template string. Replace `llm` with `llm_tier` (map `"{heavy_model}"` to `heavy`, etc.). Rename `max_iter` to `max_iterations` and `max_execution_time` to `timeout_seconds`. Add a `tools` list based on the agent's role (the tool names must match keys that will be defined in the `TOOL_REGISTRY` from section 05).

## Integration with Graph Builders

Graph builders (sections 06-08) consume the config system as follows:

```python
def build_research_graph(
    config_watcher: ConfigWatcher,
    checkpointer: AsyncPostgresSaver,
) -> CompiledStateGraph:
    configs = config_watcher.get_all_configs()
    researcher_cfg = configs["quant_researcher"]
    researcher_llm = get_chat_model(researcher_cfg.llm_tier)
    # ... bind llm and tools to node functions
```

Runners (section 11) own the `ConfigWatcher` lifecycle:

```python
async def async_main() -> None:
    watcher = ConfigWatcher(yaml_path, watch=is_dev_mode())
    watcher.register_sighup_handler()
    try:
        while not shutdown.requested:
            watcher.apply_pending_reload()
            graph = build_research_graph(watcher, checkpointer)
            await asyncio.wait_for(graph.ainvoke(state, config), timeout=budget)
    finally:
        watcher.stop()
```

## Error Handling

- **Invalid YAML at startup**: `load_agent_configs()` raises `ValueError`. The runner fails to start. This is correct -- running with invalid config is worse than not running.
- **Invalid YAML during hot-reload**: The file watcher catches the parse error, logs a warning, and does NOT set `_pending_configs`. The active config remains unchanged. The next valid edit will succeed.
- **Missing agent name**: `get_config("nonexistent")` raises `KeyError`. Graph builders should fail fast at build time, not at node execution time.
- **SIGHUP on non-Unix**: `signal.SIGHUP` does not exist on Windows. The `register_sighup_handler()` method should catch `AttributeError` and log that SIGHUP reload is unavailable. File-watch mode works on all platforms.
