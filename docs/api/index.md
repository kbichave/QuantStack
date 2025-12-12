# API Reference

This section provides API reference documentation for the QuantCore packages.

## Packages

### quantcore

The core quantitative trading library.

| Module | Description |
|--------|-------------|
| `quantcore.features` | 200+ technical indicators |
| `quantcore.backtesting` | Event-driven backtesting engine |
| `quantcore.models` | ML model training and prediction |
| `quantcore.rl` | Reinforcement learning agents |
| `quantcore.options` | Options pricing and Greeks |
| `quantcore.microstructure` | Order book and execution |
| `quantcore.research` | Statistical tests and analysis |
| `quantcore.risk` | Position sizing and controls |
| `quantcore.validation` | Leakage detection |

**Quick Import:**
```python
import quantcore as qc

# Access timeframes
qc.Timeframe.DAILY
qc.Timeframe.HOURLY_4

# Common types
qc.OHLCV
qc.Signal
qc.Returns
```

### quant_pod

Multi-agent trading system.

| Module | Description |
|--------|-------------|
| `quant_pod.crews` | Agent crew assembly |
| `quant_pod.flows` | Trading workflows |
| `quant_pod.knowledge` | Persistent knowledge store |
| `quant_pod.memory` | Agent communication |
| `quant_pod.tools` | MCP bridge and utilities |

**Quick Import:**
```python
from quant_pod.crews import TradingCrew
from quant_pod.flows import TradingDayFlow
from quant_pod.knowledge import KnowledgeStore
```

### quant_arena

Historical simulation engine.

| Module | Description |
|--------|-------------|
| `quant_arena.historical` | Simulation components |

**Quick Import:**
```python
from quant_arena.historical import Engine, SimBroker, DataLoader
```

### etrade_mcp

E-Trade MCP server.

| Module | Description |
|--------|-------------|
| `etrade_mcp.client` | E-Trade API client |
| `etrade_mcp.auth` | OAuth authentication |
| `etrade_mcp.models` | Data models |
| `etrade_mcp.server` | MCP server |

## Type Definitions

```python
from quantcore import (
    OHLCV,          # DataFrame with OHLCV columns
    Signal,         # Series with values in [-1, 1]
    Returns,        # Series of returns
    FeatureMatrix,  # DataFrame of features
    BacktestMetrics,# Dict of backtest results
    TradeRecord,    # Single trade record
)
```

## Configuration Classes

```python
from quantcore.config import (
    Timeframe,      # Enum of supported timeframes
    Settings,       # Global settings
    TIMEFRAME_PARAMS,  # Timeframe parameters
)
```

## Generating Full API Docs

For complete API documentation, use pdoc or sphinx:

```bash
# Generate with pdoc
pdoc packages/quantcore -o docs/api/quantcore

# Generate with sphinx
cd docs && sphinx-apidoc -o api ../packages/quantcore
```
