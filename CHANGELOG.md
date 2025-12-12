# Changelog

All notable changes to QuantCore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial open source release
- Comprehensive type hints with `_typing.py` module
- NumPy-style docstrings for all public APIs
- Pre-commit hooks for code quality
- GitHub Actions CI/CD pipeline
- MkDocs documentation site

### Changed
- Restructured to `packages/` monorepo layout with quantcore, quant_arena, quant_pod, etrade_mcp
- Renamed package from `trader` to `quantcore`
- Consolidated `math_models`, `microstructure`, `research`, `signals` into main quantcore package

## [0.1.0] - 2024-12-04

### Added

#### Core Framework
- Multi-timeframe hierarchical trading system
- Event-driven backtesting engine with realistic costs
- Configurable position sizing and risk management
- DuckDB-based data storage and retrieval

#### Feature Engineering
- 200+ technical indicators across categories:
  - Trend: EMA, SMA, MACD, ADX, Aroon
  - Momentum: RSI, Stochastic, Williams %R, CCI
  - Volatility: ATR, Bollinger Bands, Keltner Channels
  - Volume: OBV, VWAP, Volume Profile
  - Market Structure: Support/Resistance, Swing Points
  - Gann Analysis: Swing points, retracements
  - Wave Analysis: Elliott Wave detection
- Multi-timeframe feature factory
- Feature scaling and normalization utilities

#### Machine Learning
- LightGBM, XGBoost, CatBoost model training
- Ensemble model with weighted averaging
- SHAP-based feature importance
- Hyperparameter tuning with Optuna

#### Reinforcement Learning (Optional)
- PPO and DQN agents for trading
- Custom Gymnasium environments
- Multi-objective reward shaping
- Experience replay and prioritized sampling

#### Research Tools
- Statistical tests (ADF, Granger causality)
- Alpha decay analysis
- Walk-forward validation
- Harvey-Liu multiple testing correction
- Leakage detection utilities

#### Market Microstructure
- Limit order book simulation
- Price impact models (Almgren-Chriss, Bouchaud)
- Execution algorithms (TWAP, VWAP, IS)
- Market making simulation (Avellaneda-Stoikov)

#### Mathematical Models
- Geometric Brownian Motion
- Stochastic volatility (Heston, SABR)
- Kalman and particle filters
- Portfolio optimization utilities

#### Strategies
- Mean reversion (z-score based)
- Momentum (multi-timeframe)
- Composite strategy framework
- Options strategies (covered calls, spreads)

#### Data Sources
- Alpha Vantage integration
- FRED economic data
- Earnings calendar
- News sentiment (extensible)

### Infrastructure
- Pydantic-based configuration
- Loguru structured logging
- Comprehensive test suite
- Type hints throughout

---

## Version History

- `0.1.0` - Initial release

[Unreleased]: https://github.com/kbichave/QuantStack/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kbichave/QuantStack/releases/tag/v0.1.0

