# Contributing Guide

Thank you for your interest in contributing to QuantStack! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- Virtual environment tool (venv, conda)

### Clone and Install

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/kbichave/QuantStack.git
cd QuantStack

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode with all dependencies
pip install -e ".[all]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Repository Structure

```
QuantStack/
├── packages/           # Python packages
│   ├── quantcore/     # Core library
│   ├── quant_pod/     # Multi-agent system
│   ├── quant_arena/   # Simulation engine
│   └── etrade_mcp/    # E-Trade integration
├── configs/           # Configuration files
├── scripts/           # Utility scripts
├── tests/             # Test suite
├── docs/              # Documentation
└── examples/          # Example applications
```

## Code Style

### Formatting

We use `ruff` for linting and formatting:

```bash
# Check code
ruff check packages/

# Auto-fix issues
ruff check packages/ --fix

# Format code
ruff format packages/
```

### Type Hints

All code should include type hints:

```python
# Good
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    ...

# Bad
def calculate_rsi(prices, period=14):
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def compute_indicator(
    data: pd.DataFrame,
    indicator: str,
    **kwargs
) -> pd.Series:
    """Compute a technical indicator.
    
    Args:
        data: OHLCV DataFrame with columns [open, high, low, close, volume].
        indicator: Name of the indicator (e.g., 'RSI', 'MACD').
        **kwargs: Indicator-specific parameters.
    
    Returns:
        Series containing the indicator values.
    
    Raises:
        ValueError: If indicator is not recognized.
    
    Example:
        >>> rsi = compute_indicator(data, 'RSI', period=14)
    """
```

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/unit/test_features_base.py -v

# With coverage
pytest tests/ --cov=packages/quantcore --cov-report=html

# Run only fast tests
pytest tests/ -v -m "not slow"
```

### Writing Tests

Place tests in the appropriate directory:

- `tests/unit/` - Unit tests for individual functions
- `tests/integration/` - Integration tests
- `tests/property/` - Property-based tests (Hypothesis)

Example test:

```python
# tests/unit/test_my_module.py
import pytest
from quantcore.features import TechnicalIndicators

class TestTechnicalIndicators:
    def test_rsi_returns_series(self, sample_ohlcv):
        ti = TechnicalIndicators()
        result = ti.rsi(sample_ohlcv, period=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)
    
    def test_rsi_bounds(self, sample_ohlcv):
        ti = TechnicalIndicators()
        result = ti.rsi(sample_ohlcv, period=14)
        assert result.min() >= 0
        assert result.max() <= 100
    
    @pytest.mark.parametrize("period", [7, 14, 21])
    def test_rsi_periods(self, sample_ohlcv, period):
        ti = TechnicalIndicators()
        result = ti.rsi(sample_ohlcv, period=period)
        assert not result.isna().all()
```

### Test Fixtures

Common fixtures are in `tests/conftest.py`:

```python
# tests/conftest.py
import pytest
import pandas as pd

@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data."""
    return pd.DataFrame({
        'open': [...],
        'high': [...],
        'low': [...],
        'close': [...],
        'volume': [...]
    })
```

## Pull Request Process

### 1. Create a Branch

```bash
# Update main
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/my-new-feature
```

### 2. Make Changes

- Write code with tests
- Update documentation if needed
- Run linting and tests locally

### 3. Commit

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add Bollinger Bands squeeze indicator

- Implement squeeze detection logic
- Add unit tests
- Update indicator documentation"
```

Commit message format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `test:` Adding tests
- `refactor:` Code refactoring
- `perf:` Performance improvement

### 4. Push and Create PR

```bash
git push origin feature/my-new-feature
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- Link to related issues
- Test plan

## Adding New Features

### Adding a Technical Indicator

1. Add implementation in `packages/quantcore/features/`:

```python
# packages/quantcore/features/technical_indicators.py

def my_indicator(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate my custom indicator.
    
    Args:
        data: OHLCV DataFrame.
        period: Lookback period.
    
    Returns:
        Indicator values.
    """
    # Implementation
    return result
```

2. Add tests in `tests/unit/`:

```python
# tests/unit/test_my_indicator.py

def test_my_indicator_calculation():
    ti = TechnicalIndicators()
    result = ti.my_indicator(sample_data, period=14)
    assert ...
```

3. Update documentation if significant.

### Adding a Strategy

1. Create strategy class:

```python
# packages/quantcore/strategy/my_strategy.py

from quantcore.strategy.base import StrategyBase

class MyStrategy(StrategyBase):
    def __init__(self, param1: float = 1.0):
        self.param1 = param1
    
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        # Signal generation logic
        return signals
```

2. Add to `__init__.py` exports.
3. Write tests.

### Adding MCP Tools

1. Add tool to server:

```python
# packages/quantcore/mcp/server.py

@mcp.tool()
def my_new_tool(param: str) -> dict:
    """Description of what this tool does."""
    return {"result": process(param)}
```

2. Add tests for the tool.
3. Update MCP documentation.

## Documentation

### Building Docs Locally

```bash
# If using mkdocs (not currently set up)
mkdocs serve

# Or generate API docs with pdoc
pdoc packages/quantcore -o docs/api/generated
```

### Documentation Style

- Use clear, concise language
- Include code examples
- Keep examples runnable
- Link to related sections

## Getting Help

- Open an issue for bugs or feature requests
- Tag maintainers for review
