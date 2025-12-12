# QuantArena Architecture

QuantArena is a historical simulation engine for backtesting multi-agent trading systems with realistic execution modeling.

## Overview

QuantArena provides:
- Time-synchronized historical replay of market data
- Simulated broker with realistic execution and slippage
- Multi-asset universe management
- Integration with QuantPod agents and QuantCore analytics

## Package Structure

```
packages/quant_arena/
├── __init__.py
└── historical/
    ├── __init__.py
    ├── clock.py        # Simulation clock management
    ├── config.py       # Arena configuration
    ├── data_loader.py  # Historical data loading
    ├── engine.py       # Main simulation engine
    ├── run.py          # CLI runner
    ├── sim_broker.py   # Simulated broker
    └── universe.py     # Asset universe management
```

## Core Components

### Simulation Engine

The central orchestrator for historical simulations:

```python
from quant_arena.historical import Engine, ArenaConfig

config = ArenaConfig(
    start_date="2023-01-01",
    end_date="2023-12-31",
    initial_capital=100_000,
    symbols=["SPY", "QQQ", "IWM"],
    timeframe="1h"
)

engine = Engine(config)

# Register strategy or agents
engine.register_strategy(my_strategy)

# Run simulation
results = engine.run()
```

### Simulation Clock

Manages time progression during backtests:

```python
from quant_arena.historical import SimulationClock

clock = SimulationClock(
    start=datetime(2023, 1, 1, 9, 30),
    end=datetime(2023, 12, 31, 16, 0),
    step=timedelta(hours=1)
)

while clock.has_next():
    current_time = clock.advance()
    # Process bar at current_time
```

**Clock Features:**
- Configurable time step (tick, minute, hour, day)
- Market hours awareness (skip non-trading periods)
- Event scheduling (earnings, dividends, splits)

### Simulated Broker

Realistic order execution simulation:

```python
from quant_arena.historical import SimBroker, Order

broker = SimBroker(
    initial_capital=100_000,
    slippage_model="percentage",  # or "fixed", "volume"
    slippage_bps=5.0,
    commission_per_trade=1.0
)

# Submit order
order = Order(
    symbol="SPY",
    side="buy",
    quantity=100,
    order_type="market"
)

fill = broker.submit(order, current_bar)
print(f"Filled at ${fill.price:.2f}")
```

**Broker Features:**
- Market and limit orders
- Configurable slippage models
- Commission modeling
- Position tracking
- P&L calculation

### Data Loader

Historical data management:

```python
from quant_arena.historical import DataLoader

loader = DataLoader(
    data_source="duckdb",
    db_path="data/market_data.duckdb"
)

# Load data for simulation
data = loader.load(
    symbols=["SPY", "QQQ"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    timeframe="1h"
)

# Iterator interface
for bar in loader.iter_bars(symbols=["SPY"]):
    process(bar)
```

### Universe Management

Multi-asset simulation support:

```python
from quant_arena.historical import Universe

universe = Universe(
    symbols=["SPY", "QQQ", "IWM", "DIA"],
    sector_map={
        "SPY": "broad_market",
        "QQQ": "technology",
        "IWM": "small_cap",
        "DIA": "industrials"
    }
)

# Filter by criteria
tech_symbols = universe.filter(sector="technology")

# Dynamic universe (e.g., S&P 500 constituents)
universe.load_constituents(
    index="SPX",
    as_of_date="2023-06-01"
)
```

## Simulation Modes

### Single Strategy Backtest

```python
from quant_arena.historical import Engine
from quantcore.strategy import MeanReversionStrategy

engine = Engine(config)
strategy = MeanReversionStrategy(zscore_threshold=2.0)

engine.register_strategy(strategy)
results = engine.run()
```

### Multi-Agent Simulation

Run QuantPod agents in historical simulation:

```python
from quant_arena.historical import Engine
from quant_pod.crews import TradingCrew

engine = Engine(config)

# Register QuantPod crew
crew = TradingCrew(mode="backtest")
engine.register_agent_crew(crew)

# Run with agent decision-making
results = engine.run(
    agent_decision_interval=timedelta(hours=4)
)
```

### Strategy Competition (QuantArena)

Pit multiple strategies against each other:

```python
from quant_arena.historical import Arena

arena = Arena(config)

# Register competing strategies
arena.add_contestant("momentum", MomentumStrategy())
arena.add_contestant("mean_rev", MeanReversionStrategy())
arena.add_contestant("ml_model", MLStrategy(model_path="model.pkl"))

# Run competition
leaderboard = arena.compete()
print(leaderboard.to_dataframe())
```

## Execution Models

### Slippage Models

```python
from quant_arena.historical import SlippageModel

# Percentage-based (default)
slippage = SlippageModel(
    model_type="percentage",
    bps=5.0
)

# Volume-based (market impact)
slippage = SlippageModel(
    model_type="volume",
    impact_coefficient=0.1,
    daily_volume_pct=0.01
)

# Fixed spread
slippage = SlippageModel(
    model_type="fixed",
    spread=0.01
)
```

### Fill Models

```python
from quant_arena.historical import FillModel

# Immediate fill at slipped price
fill_model = FillModel(
    fill_type="immediate",
    partial_fills=False
)

# Probabilistic fill based on volume
fill_model = FillModel(
    fill_type="volume_weighted",
    fill_probability_fn=lambda order, bar: min(1.0, order.qty / bar.volume)
)
```

## Results and Analytics

### Performance Metrics

```python
results = engine.run()

# Access metrics
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
print(f"Win Rate: {results.win_rate:.2%}")
print(f"Profit Factor: {results.profit_factor:.2f}")

# Trade-level analysis
trades = results.trades
print(f"Total Trades: {len(trades)}")
print(f"Avg Trade Duration: {trades.duration.mean()}")
```

### Equity Curve

```python
# Get equity curve
equity = results.equity_curve

# Plot with drawdowns
from quant_arena.historical import plot_equity
plot_equity(equity, show_drawdowns=True)
```

### Trade Journal

```python
# Detailed trade log
journal = results.journal

for trade in journal:
    print(f"{trade.entry_time}: {trade.side} {trade.symbol}")
    print(f"  Entry: ${trade.entry_price:.2f}")
    print(f"  Exit: ${trade.exit_price:.2f}")
    print(f"  P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.2%})")
```

## Integration with UI

QuantArena includes a web UI for visualization:

```
examples/historical_quant_arena_ui/
├── backend/
│   └── api.py          # FastAPI backend
└── frontend/
    ├── app.py          # Streamlit main app
    └── pages/
        └── 1_Strategy_Leaderboard.py
```

### Running the UI

```bash
# Start backend
cd examples/historical_quant_arena_ui
uvicorn backend.api:app --port 8000

# Start frontend (separate terminal)
streamlit run frontend/app.py
```

## Configuration

```yaml
# configs/quant_arena.yaml
simulation:
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  initial_capital: 100000
  timeframe: "1h"

broker:
  slippage_model: "percentage"
  slippage_bps: 5.0
  commission_per_trade: 1.0
  
universe:
  symbols:
    - SPY
    - QQQ
    - IWM
  dynamic: false

execution:
  fill_model: "immediate"
  partial_fills: false
```

## CLI Usage

```bash
# Run backtest
python -m quant_arena.historical.run \
    --config configs/quant_arena.yaml \
    --strategy momentum \
    --output results/

# Run arena competition
python -m quant_arena.historical.run \
    --mode arena \
    --strategies momentum,mean_rev,ml \
    --output results/competition/
```

## Performance Optimization

### Data Caching

```python
# Pre-load all data into memory
engine = Engine(config, preload_data=True)

# Use memory-mapped files for large datasets
loader = DataLoader(
    data_source="mmap",
    mmap_path="data/market_data.mmap"
)
```

### Parallel Backtests

```python
from quant_arena.historical import parallel_backtest

# Run parameter sweep in parallel
param_grid = {
    "zscore_threshold": [1.5, 2.0, 2.5],
    "lookback": [20, 50, 100]
}

results = parallel_backtest(
    strategy_class=MeanReversionStrategy,
    param_grid=param_grid,
    config=config,
    n_jobs=4
)
```
