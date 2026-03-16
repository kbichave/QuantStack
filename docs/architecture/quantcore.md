# QuantCore Library Architecture

QuantCore is the foundational quantitative trading library providing technical analysis, backtesting, machine learning, and risk management capabilities.

## Module Overview

```
packages/quantcore/
├── analysis/          # Hyperparameter tuning, Monte Carlo, reporting
├── analytics/         # External library adapters (ffn)
├── backtesting/       # Event-driven backtesting engine
├── config/            # Settings, timeframes, configuration
├── core/              # Calendar, errors, base config
├── data/              # Data fetching and storage (DuckDB)
├── equity/            # Equity-specific strategies and pipelines
├── execution/         # Production execution: broker abstraction, risk gate, kill switch, TCA, smart routing
├── features/          # 200+ technical indicators
├── hierarchy/         # Multi-timeframe alignment, regime detection
├── labeling/          # Event and wave labeling for ML
├── math/              # Stochastic processes, filters, optimization
├── portfolio/         # Mean-variance portfolio optimization with covariance shrinkage
├── mcp/               # MCP server for AI agent access
├── microstructure/    # Order book, impact models, execution algos
├── options/           # Options pricing, Greeks, IV surface
├── research/          # Statistical tests, alpha decay, walkforward
├── risk/              # Position sizing, controls, stress testing
├── rl/                # Reinforcement learning agents
├── signals/           # Signal generation and evaluation
├── strategy/          # Strategy base classes and rules
├── utils/             # Formatting utilities
├── validation/        # Input validation, leakage detection
└── visualization/     # Plotting and trade animation
```

## Core Modules

### features/ - Technical Indicators

200+ technical indicators organized by category:

```python
from quantcore.features import TechnicalIndicators, MarketStructure

# Compute indicators
ti = TechnicalIndicators(timeframe=Timeframe.DAILY)
features = ti.compute_all(ohlcv_df)

# Market structure detection
ms = MarketStructure()
swings = ms.detect_swing_points(ohlcv_df)
```

**Indicator Categories:**
- **Trend**: EMA, SMA, MACD, ADX, Aroon, Supertrend, VWMA
- **Momentum**: RSI, Stochastic, Williams %R, CCI, ROC, MFI
- **Volatility**: ATR, Bollinger Bands, Keltner Channels, Donchian
- **Volume**: OBV, VWAP, Volume Profile, A/D Line, CMF
- **Market Structure**: Support/Resistance, Swing Points, HH/HL/LH/LL
- **Advanced**: Elliott Wave, Gann, Sentiment indicators

### backtesting/ - Backtest Engine

Event-driven backtesting with realistic cost modeling:

```python
from quantcore.backtesting import BacktestEngine, TransactionCosts

engine = BacktestEngine(
    initial_capital=100_000,
    costs=TransactionCosts(
        spread_bps=2.0,
        commission_per_share=0.005,
        slippage_bps=1.0
    )
)

result = engine.run(data=features, strategy=my_strategy)
print(f"Sharpe: {result.sharpe_ratio:.2f}")
print(f"Max DD: {result.max_drawdown:.1%}")
```

**Key Components:**
- `BacktestEngine`: Core event-driven engine
- `RealisticEngine`: Enhanced with market impact
- `OptionsEngine`: Options-specific backtesting
- `TransactionCosts`: Spread, commission, slippage modeling

### models/ - Machine Learning

Native integration with gradient boosting and ensemble methods:

```python
from quantcore.models import EnsembleModel, ModelConfig

model = EnsembleModel(
    estimators=['lightgbm', 'xgboost', 'catboost'],
    weights=[0.4, 0.3, 0.3]
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
importance = model.feature_importance(method='shap')
```

**Supported Models:**
- LightGBM, XGBoost, CatBoost
- Ensemble with weighted averaging
- SHAP-based feature importance
- Hyperparameter tuning with Optuna

### rl/ - Reinforcement Learning (Production)

Production RL system with shadow mode, online learning, and a formal promotion gate:

```python
from quantcore.rl.config import RLProductionConfig
from quantcore.rl.shadow_mode import ShadowMode
from quantcore.rl.promotion_gate import PromotionGate

# Config is env-driven (QUANTRL_ prefix) and versioned
cfg = RLProductionConfig.load()

# All agents start in shadow mode — recommendations tagged [SHADOW]
# and collected for promotion validation without affecting execution
shadow = ShadowMode(cfg)
rec = shadow.get_recommendation(symbol="SPY", direction="LONG", confidence=0.72)
# rec.tag == "[SHADOW – not yet validated]"

# Promote after sufficient observations
gate = PromotionGate(cfg)
result = gate.validate(agent_id="sizing_v1")  # walk-forward + Monte Carlo + drawdown
```

**Components:**
- `RLProductionConfig`: versioned, env-driven config (`QUANTRL_` prefix); feature flags per agent tier; shadow mode on by default
- `KnowledgeStoreRLBridge`: maps backtest results and trade history into RL training datasets
- State vector modules for execution (8 features), sizing (10 features), and meta (36 features) tiers
- `OnlineAdapter`: live trade outcome → real-time weight update with catastrophic forgetting guard
- `PromotionGate`: shadow → live validation — walk-forward folds (≥60% positive), Monte Carlo (p < 0.05), max drawdown (≤12%), Sharpe threshold (≥0.5)
- `ShadowMode`: recommendations collected as observations without changing execution
- Minimum shadow period: 63 trading days for execution/sizing agents; 126 days for meta agents

### options/ - Options Pricing

Black-Scholes pricing with Greeks and IV surface:

```python
from quantcore.options import BlackScholes, IVSurface

# Single option pricing
bs = BlackScholes(S=100, K=105, T=0.25, r=0.05, sigma=0.2)
print(f"Call: ${bs.call_price:.2f}")
print(f"Delta: {bs.delta:.3f}")

# IV surface
surface = IVSurface(option_chain)
iv = surface.interpolate(strike=105, expiry=30)
```

**Capabilities:**
- Black-Scholes with dividends
- Greeks: delta, gamma, theta, vega, rho
- Implied volatility solver
- Adapters: FinancePy, QuantLib, vollib

### microstructure/ - Market Microstructure

Order book simulation, execution algorithms, and real-time tick-data signal extraction:

```python
from quantcore.microstructure import OrderBook, TWAPExecutor
from quantcore.microstructure.microstructure_features import MicrostructureFeatureEngine

# Limit order book
lob = OrderBook()
lob.add_order(side='bid', price=99.50, size=1000)

# TWAP execution
executor = TWAPExecutor(total_quantity=10000, duration=3600)
schedule = executor.generate_schedule()

# Tick-data microstructure signals (streaming)
engine = MicrostructureFeatureEngine(symbol="SPY")
engine.on_trade(trade_tick)   # updates OFI, Kyle's Lambda, VPIN
engine.on_quote(quote_tick)   # updates bid-ask spread stats
features = engine.get_features()
# features.ofi_normalized, features.kyle_lambda, features.vpin, features.roll_spread
```

**Components:**
- `OrderBook`: Limit order book with matching engine
- `TWAPExecutor`, `VWAPExecutor`: Execution algorithms
- `ImpactModel`: Square-root price impact (Almgren-Chriss)
- `VolumeProfile`: Intraday volume distribution
- `MicrostructureFeatureEngine`: OFI, Kyle's Lambda, bid-ask spread, VPIN, trade intensity, Roll spread from tick data
- `OrderBookReconstructor`: reconstructs LOB state from `TradeTick` + `QuoteTick` streams

### research/ - Research Tools

Statistical analysis and validation:

```python
from quantcore.research import stat_tests, alpha_decay, walkforward

# Stationarity test
adf = stat_tests.adf_test(spread_series)
print(f"ADF p-value: {adf.p_value:.4f}")

# Alpha decay analysis
decay = alpha_decay.compute_decay_curve(signal, returns)
print(f"Half-life: {decay.half_life} bars")

# Walk-forward validation
results = walkforward.run(
    data=features,
    model=model,
    train_window=252,
    test_window=63
)
```

**Tools:**
- ADF, KPSS, Granger causality tests
- Alpha decay and information coefficient
- Walk-forward and purged cross-validation
- Harvey-Liu multiple testing correction

### portfolio/ - Portfolio Optimization

Markowitz mean-variance optimization with institutional-grade constraints:

```python
from quantcore.portfolio.optimizer import MeanVarianceOptimizer, OptimizationObjective

optimizer = MeanVarianceOptimizer(
    objective=OptimizationObjective.MAX_SHARPE,
    min_weight=0.0,
    max_weight=0.15,
    turnover_penalty=0.001,
)

result = optimizer.optimize(
    expected_returns=signal_series,
    price_history=ohlcv_df,
    current_weights=portfolio.weights,
)
# result.target_weights, result.expected_sharpe, result.required_trades
```

**Features:**
- Objectives: `MAX_SHARPE`, `MIN_VARIANCE`, `RISK_PARITY`, `MAX_DIVERSIFICATION`
- Ledoit-Wolf covariance shrinkage (reduces estimation error on small samples)
- Per-position weight bounds (supports long/short via negative min_weight)
- Sector-level allocation constraints
- Turnover cost penalty to prevent excessive churn
- Automatic fallback to equal-weight on infeasible constraints or SLSQP convergence failure

### hierarchy/ - Multi-Timeframe Analysis

Regime detection and timeframe alignment:

```python
from quantcore.hierarchy import RegimeClassifier, TrendFilter

# Regime detection (HMM-based)
classifier = RegimeClassifier(n_regimes=3)
regimes = classifier.fit_predict(returns)

# Higher timeframe filter
htf_filter = TrendFilter(timeframe='daily')
is_uptrend = htf_filter.is_bullish(ohlcv_4h)
```

**Components:**
- `RegimeClassifier`: HMM and changepoint detection
- `TrendFilter`: Multi-timeframe trend alignment
- `WaveContext`: Elliott Wave context
- `SwingContext`: Swing high/low detection

### risk/ - Risk Management

Position sizing and controls:

```python
from quantcore.risk import PositionSizer, RiskControls

# Kelly-based sizing
sizer = PositionSizer(method='kelly', max_risk_pct=0.02)
size = sizer.calculate(win_rate=0.55, avg_win=100, avg_loss=80)

# Risk controls
controls = RiskControls(
    max_drawdown=0.15,
    max_position_pct=0.10,
    daily_loss_limit=0.03
)
controls.check(portfolio_state)
```

**Features:**
- Kelly criterion, fixed fractional, volatility-based sizing
- Drawdown and daily loss limits
- Options-specific risk (Greeks exposure)
- SPAN margin approximation

### validation/ - Data Validation

Leakage detection and input validation:

```python
from quantcore.validation import LeakageDetector, validate_features

# Check for data leakage
detector = LeakageDetector()
issues = detector.scan(X_train, X_test, feature_names)

# Input validation
validate_features(df, required_cols=['open', 'high', 'low', 'close'])
```

**Checks:**
- Future data leakage detection
- Feature correlation analysis
- Orthogonalization for factor models
- Purged cross-validation splits

## Type Definitions

```python
from quantcore import OHLCV, Signal, Returns, FeatureMatrix

# Type aliases for clarity
data: OHLCV  # DataFrame with open, high, low, close, volume
signal: Signal  # Series with values in [-1, 1]
returns: Returns  # Series of log or simple returns
features: FeatureMatrix  # DataFrame of computed features
```

## Configuration

### `Settings` — Environment-Driven Config

`packages/quantcore/config/settings.py` uses pydantic-settings. All values are overridable via environment variables or `.env` file.

```python
from quantcore.config.settings import Settings, AlpacaSettings, IBKRSettings

settings = Settings()

# Nested provider configs — each reads its own env prefix
# ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER
print(settings.alpaca.paper)       # True (default)

# IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID
print(settings.ibkr.port)          # 4001 (IB Gateway default)

# Data provider priority (comma-separated; skips providers with missing credentials)
print(settings.data_provider_priority)  # "alpaca,polygon,alpha_vantage"
```

Key fields:

| Field | Default | Env var | Notes |
|-------|---------|---------|-------|
| `data_provider_priority` | `alpaca,polygon,alpha_vantage` | `DATA_PROVIDER_PRIORITY` | Left-to-right fallback |
| `data_end_date` | today | `DATA_END_DATE` | Defaults to today; no longer hardcoded |
| `alpaca.paper` | `true` | `ALPACA_PAPER` | Paper mode toggle |
| `ibkr.port` | `4001` | `IBKR_PORT` | 4001 = IB Gateway, 7497 = TWS |

### `Timeframe` — Extended Hierarchy

```python
from quantcore.config.timeframes import Timeframe, TIMEFRAME_HIERARCHY

# Supported timeframes (coarsest to finest)
# W1 → D1 → H4 → H1 → M30 → M15 → M5 → M1 → S5

tf = Timeframe.M5   # 5-minute bars
tf = Timeframe.S5   # 5-second bars (HFT / order flow)
```

Intraday timeframes (`M30` through `S5`) were added for microstructure-driven strategies. Each has tuned `TimeframeParams` (EMA 9/21, shorter RSI/ATR periods at M5 and below).

## Extension Points

QuantCore is designed for extensibility:

1. **Custom Indicators**: Subclass `IndicatorBase` in features/
2. **Custom Strategies**: Implement `StrategyBase` interface
3. **Custom Models**: Add adapters in models/adapters/
4. **Custom Execution**: Extend execution algorithms in microstructure/

## Performance Considerations

- **Vectorized Operations**: All indicators use NumPy/Pandas vectorization
- **Lazy Loading**: Heavy modules loaded on-demand
- **Caching**: DuckDB for efficient data caching
- **Parallel Processing**: Support for joblib parallelization
