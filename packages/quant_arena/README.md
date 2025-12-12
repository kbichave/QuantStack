# QuantArena

Historical simulation engine for backtesting multi-agent trading systems with realistic execution modeling.

## Installation

QuantArena is part of the main repository:

```bash
uv sync --all-extras
```

## Components

| Component | Description |
|-----------|-------------|
| `historical/clock.py` | Simulation clock with market hours awareness |
| `historical/sim_broker.py` | Simulated broker with slippage and commission |
| `historical/data_loader.py` | Historical data loading from DuckDB |
| `historical/config.py` | Arena configuration |
| `historical/run.py` | CLI runner |

## Quick Usage

```python
from quant_arena.historical import Engine, ArenaConfig

config = ArenaConfig(
    start_date="2023-01-01",
    end_date="2023-12-31",
    initial_capital=100_000,
    symbols=["SPY", "QQQ"],
    timeframe="1h"
)

engine = Engine(config)
engine.register_strategy(my_strategy)
results = engine.run()
```

## CLI Usage

```bash
python -m quant_arena.historical.run \
    --symbols SPY,QQQ \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --equity 100000
```

## Web UI

QuantArena includes a web UI for visualization:

```bash
# Start backend
uvicorn examples.historical_quant_arena_ui.backend.api:app --port 8000

# Start frontend
streamlit run examples/historical_quant_arena_ui/frontend/app.py
```

## Documentation

See [Architecture Documentation](../../docs/architecture/quant_arena.md) for detailed component descriptions.
