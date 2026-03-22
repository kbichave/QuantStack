# API Reference

This section provides API reference documentation for the QuantStack package.

## Package: quantstack

The unified quantitative trading package at `src/quantstack/`.

### Core Library (`quantstack.core`)

| Module | Description |
|--------|-------------|
| `quantstack.core.features` | 200+ technical indicators |
| `quantstack.core.backtesting` | Event-driven backtesting engine |
| `quantstack.core.strategy` | Strategy base classes and rules |
| `quantstack.ml` | ML model training and prediction |
| `quantstack.rl` | Reinforcement learning agents |
| `quantstack.core.options` | Options pricing and Greeks |
| `quantstack.core.microstructure` | Order book and execution |
| `quantstack.core.research` | Statistical tests and analysis |
| `quantstack.core.risk` | Position sizing and controls |
| `quantstack.core.validation` | Leakage detection |

### Execution & Operations

| Module | Description |
|--------|-------------|
| `quantstack.signal_engine` | 7 concurrent Python collectors |
| `quantstack.execution` | Risk gate, order lifecycle, broker routers |
| `quantstack.coordination` | Event bus, auto-promoter, supervisor |
| `quantstack.autonomous` | Unattended trading loops |
| `quantstack.alpha_discovery` | Strategy generation |
| `quantstack.learning` | IC/ICIR tracking, drift detection |
| `quantstack.monitoring` | Signal degradation detection |
| `quantstack.flows` | Trading workflows |
| `quantstack.knowledge` | Persistent knowledge store |
| `quantstack.mcp` | Unified MCP server (120+ tools) |
| `quantstack.api` | FastAPI REST server |

**Quick Import:**
```python
from quantstack.core.features import TechnicalIndicators
from quantstack.core.config import Timeframe, Settings
from quantstack.core.backtesting import BacktestEngine
from quantstack.core.options import BlackScholes
from quantstack.core.strategy.base import StrategyBase
```

## Type Definitions

```python
from quantstack.core import (
    OHLCV,          # DataFrame with OHLCV columns
    Signal,         # Series with values in [-1, 1]
    Returns,        # Series of returns
    FeatureMatrix,  # DataFrame of features
)
```

## Configuration Classes

```python
from quantstack.core.config import (
    Timeframe,         # Enum of supported timeframes
    Settings,          # Global settings
    TIMEFRAME_PARAMS,  # Timeframe parameters
)
```

## Generating Full API Docs

```bash
# Generate with pdoc
pdoc src/quantstack -o docs/api/quantstack

# Generate with sphinx
cd docs && sphinx-apidoc -o api ../src/quantstack
```
