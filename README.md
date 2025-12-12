<p align="center">
  <img src="logo.jpg" alt="QuantStack Logo" width="200"/>
</p>

<h1 align="center">QuantStack</h1>

<p align="center">
  <strong>Unified stack: QuantCore + QuantPod + QuantArena</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <a href="docs/architecture/README.md">Architecture</a> •
  <a href="docs/guides/quickstart.md">Quick Start</a> •
  <a href="docs/api/index.md">API Reference</a> •
  <a href="https://github.com/kbichave/QuantStack/issues">Issues</a>
</p>

---

QuantStack is a comprehensive quantitative trading platform composed of:

- **QuantCore**: Core research library (features, models, backtesting, options, RL).
- **QuantArena**: Historical simulation and execution realism for backtests.
- **QuantPod**: Multi-agent trading system (CrewAI) that orchestrates strategy pods.

Together they provide end-to-end research, simulation, and agent-driven execution.

## Repository Structure

```
QuantStack/
├── packages/                    # All Python packages
│   ├── quantcore/               # Core quantitative trading library
│   ├── quant_arena/             # Historical backtesting simulation
│   ├── quant_pod/               # Multi-agent trading system (CrewAI)
│   └── etrade_mcp/              # E-Trade MCP server
├── configs/                     # Configuration files
│   ├── quantcore/               # QuantCore configs
│   └── quant_pod/               # Quant Pod configs
├── scripts/                     # Utility scripts
│   ├── data/                    # Data fetching scripts
│   ├── pipelines/               # Pipeline runners
│   └── models/                  # Model training scripts
├── examples/                    # Example applications
├── tests/                       # Test suite
└── docs/                        # Documentation
    └── architecture/            # Architecture docs
```

## Features

### Technical Indicators (200+)
- **Trend**: EMA, SMA, MACD, ADX, Aroon, Supertrend
- **Momentum**: RSI, Stochastic, Williams %R, CCI, ROC
- **Volatility**: ATR, Bollinger Bands, Keltner Channels, Donchian
- **Volume**: OBV, VWAP, Volume Profile, Accumulation/Distribution
- **Market Structure**: Support/Resistance, Swing Points, HH/HL/LH/LL
- **Advanced**: Elliott Wave detection, Gann analysis, Sentiment

### Backtesting
- Event-driven engine with configurable fills
- Transaction cost modeling (spread, slippage)
- Walk-forward validation framework
- Monte Carlo simulation
- Purged cross-validation for ML

### Machine Learning
- Native LightGBM, XGBoost, CatBoost support
- Ensemble methods with weighted averaging
- SHAP-based feature importance
- Hyperparameter tuning with Optuna
- Data leakage detection

### Reinforcement Learning (Experimental)
- PPO and DQN agent implementations
- Custom trading environments (execution, sizing, spread)
- Multi-objective reward shaping
- **Note**: RL environments are experimental; see [RL documentation](packages/quantcore/rl/README.md)

### Research Tools
- Statistical tests (ADF, Granger, regime switching)
- Alpha decay analysis
- Harvey-Liu multiple testing correction
- Information coefficient analysis

### Market Microstructure
- Limit order book simulation
- Square-root price impact model
- Execution simulation with market impact
- Kyle's lambda estimation

### Options Pricing
- Black-Scholes pricing with dividends (single options)
- Greeks: delta, gamma, theta, vega, rho
- Implied volatility solver
- **Note**: Single-option pricing only; portfolio Greeks aggregation not yet implemented

## Quick Start

### Installation

This project uses **[uv](https://github.com/astral-sh/uv)** for fast, reliable dependency management. UV is a next-generation Python package manager that is 10-100x faster than pip.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/kbichave/QuantStack.git
cd QuantStack

# Install with uv (recommended) - creates .venv automatically
uv sync --all-extras

# Alternative: pip install (not recommended)
pip install -e ".[all]"
```

> **Note**: We strongly recommend using `uv` for all development. It automatically manages virtual environments, handles dependency resolution faster, and ensures reproducible builds via `uv.lock`.

### Basic Usage

```python
import quantcore as qc

# Load data
manager = qc.DataManager()
data = manager.fetch("AAPL", timeframe=qc.Timeframe.DAILY)

# Compute features
factory = qc.FeatureFactory(timeframe=qc.Timeframe.DAILY)
features = factory.compute_all(data)

# Run backtest
result = qc.run_backtest(
    data=features,
    strategy=qc.MeanReversionStrategy(zscore_threshold=2.0),
    initial_capital=100_000,
)

print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {result['max_drawdown']:.1f}%")
print(f"Win Rate: {result['win_rate']:.1f}%")
```

### Example: WTI-Brent Spread Trading

```python
import quantcore as qc
from quantcore.research import stat_tests, alpha_decay

# Fetch commodity data
wti = manager.fetch("CL=F", timeframe=qc.Timeframe.DAILY)
brent = manager.fetch("BZ=F", timeframe=qc.Timeframe.DAILY)

# Compute spread
spread = wti["close"] - brent["close"]

# Test for mean reversion (stationarity)
adf_result = stat_tests.adf_test(spread)
print(f"ADF p-value: {adf_result.p_value:.4f}")  # < 0.05 = stationary

# Analyze alpha decay
decay = alpha_decay.compute_decay_curve(signal=spread_zscore, returns=returns)
print(f"Half-life: {decay['half_life']} bars")

# Run spread strategy backtest
result = qc.run_backtest(
    data=spread_df,
    strategy=SpreadMeanReversion(entry_zscore=2.0, exit_zscore=0.0),
    params={"position_size": 1000, "spread_cost": 0.05}
)
```

## Module Maturity

| Module | Status | Notes |
|--------|--------|-------|
| `features` | Stable | 200+ indicators |
| `backtesting` | Stable | Event-driven engine |
| `models` | Stable | ML integration |
| `research` | Stable | Statistical tools |
| `validation` | Stable | Leakage detection, CV |
| `microstructure` | Stable | LOB, impact models |
| `options` | Stable | Single-option pricing |
| `rl` | **Experimental** | See [RL README](packages/quantcore/rl/README.md) |

## Comparison with Alternatives

| Feature | QuantCore | Zipline | Backtrader | VectorBT |
|---------|:---------:|:-------:|:----------:|:--------:|
| Multi-timeframe | Yes | No | Partial | Partial |
| Feature Engineering | 200+ | No | Partial | Partial |
| ML Integration | Yes | No | No | Partial |
| RL Support | Experimental | No | No | No |
| Walk-Forward | Yes | No | No | Partial |
| Microstructure | Yes | No | No | No |
| Type Hints | Yes | No | No | Partial |

## Documentation

- **[Architecture Overview](docs/architecture/README.md)** - System design and components
- **[Quick Start Guide](docs/guides/quickstart.md)** - Get up and running
- **[API Reference](docs/api/index.md)** - Module documentation
- **[Contributing Guide](docs/guides/contributing.md)** - How to contribute

## Development

This project uses **[uv](https://github.com/astral-sh/uv)** for all development tasks. The `uv.lock` file ensures reproducible builds across all environments.

```bash
# Clone repository
git clone https://github.com/kbichave/QuantStack.git
cd QuantStack

# Install with uv (creates .venv automatically)
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest tests/ -v

# Run linting
uv run ruff check packages/quantcore

# Run a script
uv run python scripts/run_trading_pipeline.py --help

# Add a new dependency
uv add <package-name>

# Add a dev dependency
uv add --dev <package-name>
```

### Why UV?

- **Speed**: 10-100x faster than pip for dependency resolution and installation
- **Reproducibility**: `uv.lock` ensures everyone uses identical dependency versions
- **Simplicity**: Automatically manages virtual environments (`.venv`)
- **Compatibility**: Works with standard `pyproject.toml` and existing Python tools

## Contributing

Contributions are welcome! Please see the [Contributing Guide](docs/guides/contributing.md) for details.

- [Report bugs](https://github.com/kbichave/QuantStack/issues/new?template=bug_report.md)
- [Request features](https://github.com/kbichave/QuantStack/issues/new?template=feature_request.md)
- [Improve documentation](https://github.com/kbichave/QuantStack/tree/main/docs)
- [Submit pull requests](https://github.com/kbichave/QuantStack/pulls)

## License

QuantStack is licensed under the [Apache License 2.0](LICENSE).

## Citation

If you use QuantCore in your research, please cite:

```bibtex
@software{quantstack2024,
  title = {QuantStack: Unified Quantitative Trading Stack},
  author = {Bichave, Kshitij and Contributors},
  year = {2024},
  url = {https://github.com/kbichave/QuantStack}
}
```

## Disclaimer

This software is for educational and research purposes only. Do not use it for actual trading without proper risk management and regulatory compliance. Past performance does not guarantee future results.

---

<p align="center">
  Made with care by Kshitij Bichave
</p>
