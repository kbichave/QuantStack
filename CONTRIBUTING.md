# Contributing to QuantStack

Thank you for your interest in contributing to QuantStack! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- A GitHub account

### Types of Contributions

Contributions are welcome! Here are some ways to help:

- **Bug fixes**: Fix issues reported in GitHub Issues
- **Features**: Implement new features (please discuss in an issue first)
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve tests
- **Performance**: Optimize existing code
- **Research**: Add new trading strategies, indicators, or research modules

## Development Setup

1. **Fork the repository**

   Click the "Fork" button on GitHub to create your own copy.

2. **Clone your fork**

   ```bash
   git clone https://github.com/kbichave/QuantStack.git
   cd QuantStack
   ```

3. **Set up the development environment**

   This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install dependencies (creates .venv automatically)
   uv sync --all-extras

   # Install pre-commit hooks
   uv run pre-commit install
   ```

4. **Create a branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

## Making Changes

### Branch Naming

Use descriptive branch names:

- `feature/add-bollinger-bands` - New features
- `fix/backtest-memory-leak` - Bug fixes
- `docs/improve-api-reference` - Documentation
- `refactor/simplify-feature-factory` - Code refactoring
- `perf/optimize-zscore-calculation` - Performance improvements

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvement
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(features): add Keltner Channel indicator

fix(backtest): correct Sharpe ratio calculation for daily data

docs(readme): add installation instructions for M1 Macs
```

## Code Style

### Python Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Format code
ruff format packages/quantcore tests

# Check for issues
ruff check packages/quantcore tests

# Auto-fix issues
ruff check --fix packages/quantcore tests
```

### Type Hints

All public functions must have type hints:

```python
def calculate_rsi(
    prices: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate RSI indicator."""
    ...
```

### Docstrings

Use NumPy-style docstrings for all public functions and classes:

```python
def calculate_zscore(
    series: pd.Series,
    period: int,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Calculate rolling z-score.

    Parameters
    ----------
    series : pd.Series
        Input time series.
    period : int
        Lookback period for mean and standard deviation.
    min_periods : int, optional
        Minimum observations required. Default is period // 2.

    Returns
    -------
    pd.Series
        Z-score series.

    Examples
    --------
    >>> prices = pd.Series([100, 102, 98, 103, 97])
    >>> zscore = calculate_zscore(prices, period=3)

    Notes
    -----
    Z-scores are commonly used for mean reversion strategies.
    """
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/unit/test_features.py -v

# Run with coverage
uv run pytest tests/ --cov=packages/quantcore --cov-report=html

# Run only fast tests (exclude slow)
uv run pytest tests/ -m "not slow"
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use fixtures from `tests/conftest.py`
- Test edge cases and error conditions
- Aim for > 80% coverage on new code

Example test:

```python
import pytest
import pandas as pd
from quantcore.features.base import FeatureBase


class TestFeatureBase:
    """Tests for FeatureBase class."""

    def test_zscore_calculation(self, sample_ohlcv_df):
        """Test z-score calculation."""
        zscore = FeatureBase.zscore(sample_ohlcv_df["close"], period=20)

        assert len(zscore) == len(sample_ohlcv_df)
        assert zscore.iloc[19:].notna().all()  # After warmup
        assert abs(zscore.mean()) < 0.5  # Approximately centered

    def test_zscore_handles_zero_std(self):
        """Test z-score with constant values (zero std)."""
        constant = pd.Series([100.0] * 30)
        zscore = FeatureBase.zscore(constant, period=10)

        assert zscore.isna().all()  # Should be NaN, not inf
```

## Documentation

### Building Documentation

```bash
# Build docs locally
mkdocs serve

# View at http://127.0.0.1:8000
```

### Documentation Guidelines

- Update docstrings when changing function signatures
- Add examples for new features
- Update the changelog for user-facing changes
- Include type information in documentation

## Submitting Changes

### Pull Request Process

1. **Update your branch**

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**

   ```bash
   # Lint and format
   uv run ruff check packages/quantcore tests
   uv run ruff format --check packages/quantcore tests

   # Type check
   uv run mypy packages/quantcore

   # Tests
   uv run pytest tests/ -v
   ```

3. **Push your changes**

   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request**

   - Use a clear title following commit conventions
   - Fill out the PR template
   - Link related issues
   - Request reviews from maintainers

### PR Requirements

- All CI checks must pass
- At least one maintainer approval
- No merge conflicts
- Documentation updated if needed
- Tests added for new functionality

## Release Process

Releases are managed by maintainers. The process:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a git tag: `git tag v0.1.0`
4. Push tag: `git push origin v0.1.0`
5. GitHub Actions will build and publish to PyPI

## Questions?

- Open a [GitHub Discussion](https://github.com/kbichave/QuantStack/discussions) for questions
- Open an [Issue](https://github.com/kbichave/QuantStack/issues) for bugs or feature requests
- Check existing issues before creating new ones

Thank you for contributing!

