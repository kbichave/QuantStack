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
├── execution/         # Paper trading, slippage, costs
├── features/          # 200+ technical indicators
├── hierarchy/         # Multi-timeframe alignment, regime detection
├── labeling/          # Event and wave labeling for ML
├── math/              # Stochastic processes, filters, optimization
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

### rl/ - Reinforcement Learning

Custom trading environments and agents (experimental):

```python
from quantcore.rl.execution import ExecutionEnv, PPOAgent
from quantcore.rl.sizing import SizingEnv

# Execution optimization
env = ExecutionEnv(order_size=10000, time_horizon=60)
agent = PPOAgent(env)
agent.train(timesteps=100_000)

# Position sizing
sizing_env = SizingEnv(max_position=1.0)
```

**Environment Types:**
- `ExecutionEnv`: Optimal execution with market impact
- `SizingEnv`: Dynamic position sizing
- `SpreadEnv`: Spread trading optimization
- `OptionsEnv`: Options strategy learning

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

Order book simulation and execution algorithms:

```python
from quantcore.microstructure import OrderBook, TWAPExecutor

# Limit order book
lob = OrderBook()
lob.add_order(side='bid', price=99.50, size=1000)

# TWAP execution
executor = TWAPExecutor(total_quantity=10000, duration=3600)
schedule = executor.generate_schedule()
```

**Components:**
- `OrderBook`: Limit order book with matching engine
- `TWAPExecutor`, `VWAPExecutor`: Execution algorithms
- `ImpactModel`: Square-root price impact (Almgren-Chriss)
- `VolumeProfile`: Intraday volume distribution

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

```python
from quantcore.config import Timeframe, Settings

# Timeframe configuration
tf = Timeframe.HOURLY_4
print(tf.minutes)  # 240

# Global settings
settings = Settings()
settings.data_dir = "/path/to/data"
```

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
