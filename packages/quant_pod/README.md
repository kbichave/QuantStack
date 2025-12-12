# Quant Pod

Multi-agent trading system built on CrewAI for 24/7 market monitoring and intelligent trade execution.

## Installation

Quant Pod is part of the main repository:

```bash
uv sync --all-extras
```

## Components

| Component | Description |
|-----------|-------------|
| `crews/` | Agent crew definitions and assembly |
| `flows/` | Orchestrated trading workflows |
| `knowledge/` | Persistent knowledge and policy storage |
| `memory/` | Blackboard pattern for agent communication |
| `tools/` | MCP bridge and utility tools |
| `prompts/` | Agent prompts (ICs, Pod Managers, SuperTrader) |

## Agent Hierarchy

```
SuperTrader (Orchestrator)
    ├── Market Monitor Pod Manager
    │   ├── Market Snapshot IC
    │   └── Regime Detector IC
    ├── Technicals Pod Manager
    │   ├── Structure Levels IC
    │   ├── Trend Momentum IC
    │   └── Volatility IC
    ├── Quant Pod Manager
    │   ├── StatArb IC
    │   └── Options Vol IC
    └── Risk Pod Manager
        ├── Risk Limits IC
        └── Calendar Events IC
```

## Quick Usage

```python
from quant_pod.crews import TradingCrew

crew = TradingCrew(llm_model="gpt-4", verbose=True)

result = crew.kickoff(
    inputs={
        "symbols": ["SPY", "QQQ"],
        "date": "2024-01-15"
    }
)
```

## Configuration

Configs now ship inside the package:
- Agent prompts live in `packages/quant_pod/prompts/`
- Task definitions live in `packages/quant_pod/crews/config/tasks.yaml`
- Schedules and parameters are defined in code/flows (no external YAML required)

## Documentation

See [Architecture Documentation](../../docs/architecture/quant_pod.md) for detailed agent descriptions.
