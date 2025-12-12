# QuantCore

Institutional-grade quantitative trading research library with 200+ technical indicators, backtesting, ML/RL integration, and options pricing.

## Installation

QuantCore is part of the main repository:

```bash
uv sync --all-extras
```

## Modules

| Module | Description |
|--------|-------------|
| `features` | 200+ technical indicators (trend, momentum, volatility, volume) |
| `backtesting` | Event-driven backtesting engine with cost modeling |
| `models` | ML integration (LightGBM, XGBoost, CatBoost, ensembles) |
| `rl` | Reinforcement learning agents (experimental) |
| `options` | Black-Scholes pricing, Greeks, IV surface |
| `microstructure` | Order book simulation, impact models |
| `research` | Statistical tests, alpha decay, walk-forward |
| `risk` | Position sizing, drawdown controls |
| `validation` | Data leakage detection, purged CV |
| `hierarchy` | Multi-timeframe analysis, regime detection |

## Quick Usage

```python
import quantcore as qc
from quantcore.features import TechnicalIndicators
from quantcore.backtesting import BacktestEngine

# Compute indicators
ti = TechnicalIndicators(timeframe=qc.Timeframe.DAILY)
features = ti.compute_all(ohlcv_df)

# Run backtest
engine = BacktestEngine(initial_capital=100_000)
results = engine.run(data=features, strategy=my_strategy)

print(f"Sharpe: {results.sharpe_ratio:.2f}")
print(f"Max DD: {results.max_drawdown:.2%}")
```

## MCP Server

Start the QuantCore MCP server to expose tools to AI agents:

```bash
quantcore-mcp
```

## Documentation

See [Architecture Documentation](../../docs/architecture/quantcore.md) for detailed module descriptions.
