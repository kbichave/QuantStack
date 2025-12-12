# Quick Start Guide

Get started with QuantStack in 5 minutes.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/kbichave/QuantStack.git
cd QuantStack

# Install dependencies (creates .venv automatically)
uv sync --all-extras
```

## Basic Usage

### 1. Compute Technical Indicators

```python
import quantcore as qc
from quantcore.features import TechnicalIndicators

# Create indicator calculator
ti = TechnicalIndicators(timeframe=qc.Timeframe.DAILY)

# Load your OHLCV data (pandas DataFrame)
import pandas as pd
data = pd.read_csv("your_data.csv")

# Compute all indicators
features = ti.compute_all(data)
print(f"Computed {len(features.columns)} features")
```

### 2. Run a Backtest

```python
from quantcore.backtesting import BacktestEngine
from quantcore.strategy import MeanReversionStrategy

# Create strategy
strategy = MeanReversionStrategy(
    zscore_threshold=2.0,
    lookback_period=20
)

# Create engine
engine = BacktestEngine(
    initial_capital=100_000,
    commission=0.001  # 0.1%
)

# Run backtest
results = engine.run(data=features, strategy=strategy)

# Print results
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

### 3. Train an ML Model

```python
from quantcore.models import EnsembleModel
from quantcore.labeling import EventLabeler

# Create labels (e.g., triple barrier)
labeler = EventLabeler(
    profit_target=0.02,
    stop_loss=0.01,
    max_holding_period=10
)
labels = labeler.fit_transform(data)

# Split data
train_size = int(len(features) * 0.8)
X_train, X_test = features[:train_size], features[train_size:]
y_train, y_test = labels[:train_size], labels[train_size:]

# Train ensemble
model = EnsembleModel(
    estimators=['lightgbm', 'xgboost'],
    weights=[0.6, 0.4]
)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2%}")
```

### 4. Price Options

```python
from quantcore.options import BlackScholes

# Create option pricer
bs = BlackScholes(
    S=100,      # Current stock price
    K=105,      # Strike price
    T=0.25,     # Time to expiry (years)
    r=0.05,     # Risk-free rate
    sigma=0.20  # Volatility
)

# Get prices and Greeks
print(f"Call Price: ${bs.call_price:.2f}")
print(f"Put Price: ${bs.put_price:.2f}")
print(f"Delta: {bs.delta:.3f}")
print(f"Gamma: {bs.gamma:.4f}")
print(f"Theta: {bs.theta:.4f}")
print(f"Vega: {bs.vega:.4f}")
```

## Using MCP Servers

### Start QuantCore MCP Server

```bash
quantcore-mcp
```

### Query via CLI

```bash
# Using mcp-cli (if installed)
mcp call quantcore compute_indicator \
    --symbol SPY \
    --indicator RSI \
    --period 14
```

## Running QuantArena

### Backtest in Simulation

```python
from quant_arena.historical import Engine, ArenaConfig

config = ArenaConfig(
    start_date="2023-01-01",
    end_date="2023-12-31",
    initial_capital=100_000,
    symbols=["SPY", "QQQ"]
)

engine = Engine(config)
engine.register_strategy(your_strategy)
results = engine.run()
```

### Launch UI

```bash
# Terminal 1: Backend
uv run uvicorn examples.historical_quant_arena_ui.backend.api:app --port 8000

# Terminal 2: Frontend
uv run streamlit run examples/historical_quant_arena_ui/frontend/app.py
```

## Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Specific module
uv run pytest tests/unit/test_features_base.py -v

# With coverage
uv run pytest tests/ --cov=packages/quantcore
```

## Next Steps

- Read the [Architecture Overview](../architecture/README.md)
- Explore [QuantCore modules](../architecture/quantcore.md)
- Learn about [Multi-Agent Trading](../architecture/quant_pod.md)
- Check out [Alpha Vantage API docs](../alphavantage_api.md)
