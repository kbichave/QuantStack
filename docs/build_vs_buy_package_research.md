## Build-vs-Buy Package Research
<!-- source: ecosystem-2, confidence: medium, topic: build-vs-buy, wave: 1 -->
<!-- date: 2026-04-07 -->

Research into packages that could replace or enhance custom code in QuantStack (193k LOC).
Each area assessed for: maturity, fit, integration cost, and whether buy beats build.

---

### 1. Signal Aggregation / Factor Combination

**Current state:** Custom `signal_engine/` subsystem (8,638 LOC across 35 files). `synthesis.py` (1,005 LOC) combines signals. `ic_weights.py` (153 LOC) for information-coefficient weighting. `correlation.py` (186 LOC).

| Package | Version | Stars | License | Install | Last Active |
|---------|---------|-------|---------|---------|-------------|
| **alphalens-reloaded** | 0.4.6 | 564 | Apache-2.0 | `pip install alphalens-reloaded` | Jun 2025 |
| **empyrical-reloaded** | 0.5.12 | ~1,478 (orig) | Apache-2.0 | `pip install empyrical-reloaded` | Active |

**alphalens-reloaded** -- Performance analysis of predictive (alpha) stock factors. Computes IC, factor returns, turnover, and sector analysis. Originally Quantopian's factor evaluation toolkit.

- **QuantStack fit:** Could replace `ic_weights.py` and add rigorous factor evaluation (IC by quantile, turnover analysis, factor decay). Does NOT replace synthesis -- it evaluates factors, not combines them.
- **Integration effort:** LOW (2-3 days). Wrap around existing signal outputs. Run as offline evaluation, not in hot path.
- **Verdict: BUY for factor evaluation. Keep custom synthesis.**

**empyrical-reloaded** -- Performance/risk stats (Sharpe, Sortino, VaR, max drawdown, alpha, beta). Drop-in replacement for manual metric calculations.

- **QuantStack fit:** Already partially duplicated in custom code. Would standardize all performance metric calculations.
- **Integration effort:** LOW (1-2 days). Replace scattered metric calculations.
- **Verdict: BUY.** No reason to maintain custom Sharpe/Sortino/VaR formulas.

---

### 2. Backtesting Framework

**Current state:** Custom `core/equity/backtester.py` (163 LOC) + `backtest_tools.py`. Lightweight walk-forward engine.

| Package | Version | Stars | License | Install |
|---------|---------|-------|---------|---------|
| **vectorbt** | 0.28.5 | 7,102 | Apache-2.0 + Commons Clause | `pip install vectorbt` |
| **backtesting.py** | 0.6.5 | 8,167 | AGPL-3.0 | `pip install backtesting` |
| **bt** | 1.1.5 | 2,843 | MIT | `pip install bt` |
| **zipline-reloaded** | 3.1.1 | 1,704 | Apache-2.0 | `pip install zipline-reloaded` |
| **Lean** (QuantConnect) | -- | 18,281 | Apache-2.0 | C# engine, Python API | 

**vectorbt** -- Vectorized backtesting on pandas/NumPy, Numba-accelerated. Signal-based tooling, portfolio simulation, walk-forward optimization. The most feature-rich Python backtester.

- **QuantStack fit:** Could replace the entire custom backtester. Supports ML label generation, signal ranking, robustness testing. Walk-forward built-in.
- **Gotcha:** Commons Clause license restricts commercial selling of the library itself (not your trades). Pro version is paid. Core OSS version sufficient for QuantStack.
- **Integration effort:** MEDIUM (1-2 weeks). Need to adapt signal format to vbt's expected inputs.
- **Verdict: STRONG BUY.** The 163 LOC backtester is underbuilt for production. vectorbt adds Monte Carlo, walk-forward, combinatorial optimization for free.

**backtesting.py** -- Simple event-driven backtester with built-in optimization. Interactive HTML reports.

- **Gotcha:** AGPL-3.0 license is viral -- would require open-sourcing QuantStack if distributed. Not suitable.
- **Verdict: SKIP (license).** 

**bt** -- Tree-based strategy composition (Algo nodes). MIT license. Clean API.

- **QuantStack fit:** Good for portfolio-level backtesting (rebalancing strategies). Less suited for signal-level testing.
- **Integration effort:** MEDIUM. Different paradigm from signal-based approach.
- **Verdict: CONSIDER as complement to vectorbt for portfolio-level tests.**

**zipline-reloaded** -- Event-driven, Quantopian's engine. Most realistic (handles slippage, commissions, corporate actions).

- **QuantStack fit:** Overkill for current needs. Heavy dependency footprint. Best for institutional-grade event-driven backtests.
- **Integration effort:** HIGH (2-4 weeks). Requires Zipline data bundles, specific API patterns.
- **Verdict: SKIP for now.** vectorbt is more pragmatic.

---

### 3. Portfolio Optimization

**Current state:** `portfolio/optimizer.py` (308 LOC) + `core/portfolio/optimizer.py` (594 LOC). Two optimizer files (likely needs consolidation).

| Package | Version | Stars | License | Install |
|---------|---------|-------|---------|---------|
| **PyPortfolioOpt** | 1.6.0 | 5,622 | MIT | `pip install pyportfolioopt` |
| **Riskfolio-Lib** | 7.2.1 | 4,032 | BSD-3 | `pip install riskfolio-lib` |
| **skfolio** | 0.17.0 | 1,923 | BSD-3 | `pip install skfolio` |

**PyPortfolioOpt** -- Mean-variance, Black-Litterman, HRP, shrinkage estimators. The standard Python portfolio optimization library.

- **QuantStack fit:** Direct replacement for custom optimizer code. HRP and risk parity already built in.
- **Integration effort:** LOW-MEDIUM (3-5 days). Map existing position data to PyPortfolioOpt's expected returns + covariance inputs.
- **Verdict: STRONG BUY.** Maintains 902 LOC of tested optimizer code so you don't have to.

**Riskfolio-Lib** -- 24 risk measures, HRP/HERC, Black-Litterman, worst-case optimization. Most comprehensive.

- **QuantStack fit:** Superset of PyPortfolioOpt. Better if you need CVaR, CDaR, or worst-case robust optimization. Built on CVXPY.
- **Integration effort:** MEDIUM (1 week). More complex API than PyPortfolioOpt.
- **Verdict: BUY if you need advanced risk measures (CVaR optimization). Otherwise PyPortfolioOpt is simpler.**

**skfolio** -- Scikit-learn compatible portfolio optimization. Cross-validation, hyperparameter tuning for portfolios.

- **QuantStack fit:** The sklearn integration is compelling -- `fit()`/`predict()` API, `cross_val_score()` for portfolio strategies. Newest of the three.
- **Integration effort:** MEDIUM (1 week). Newer library, less battle-tested.
- **Verdict: WATCH.** Promising but only 1,923 stars. Consider after v1.0.

---

### 4. Options Pricing / Greeks

**Current state:** py_vollib + custom code. `pyproject.toml` does not list py_vollib as a dependency (may be vendored or removed).

| Package | Version | Stars | License | Install |
|---------|---------|-------|---------|---------|
| **QuantLib** | 1.41 | N/A (C++ core) | BSD-3 | `pip install QuantLib` |
| **FinancePy** | 1.0.1 | 2,864 | GPL-3.0 | `pip install financepy` |

**QuantLib** -- The gold standard. C++ library with Python bindings. Covers everything: options, bonds, interest rates, volatility surfaces, Monte Carlo.

- **QuantStack fit:** Overkill for equity options but unbeatable for accuracy. Handles American options, vol surfaces, exotic payoffs.
- **Gotcha:** C++ compilation can be painful. Pre-built wheels now available (v1.41, Jan 2026).
- **Integration effort:** MEDIUM (1 week). Learning curve is steep but API is well-documented.
- **Verdict: BUY for production options pricing.** Replace py_vollib. QuantLib handles American exercise correctly (py_vollib does not).

**FinancePy** -- Pure Python (Numba-compiled). Covers options, bonds, FX, credit derivatives. Fast enough for most use cases.

- **Gotcha:** GPL-3.0 license is viral. Not suitable if QuantStack is ever distributed.
- **Integration effort:** LOW (2-3 days). Simpler API than QuantLib.
- **Verdict: SKIP (GPL license).** Use QuantLib instead.

---

### 5. Regime Detection

**Current state:** `core/regime_detector.py` (100 LOC), `signal_engine/collectors/regime.py` (328 LOC), `core/hierarchy/regime_classifier.py`, `execution/regime_flip.py`. Already uses hmmlearn (in ml extras).

| Package | Version | Stars | License | Install |
|---------|---------|-------|---------|---------|
| **hmmlearn** | 0.3.3 | 3,399 | BSD | `pip install hmmlearn` |
| **pomegranate** | 1.1.2 | N/A | MIT | `pip install pomegranate` |

**hmmlearn** -- Already a dependency. Scikit-learn API for HMMs (Gaussian, GMM-HMM, Multinomial).

- **QuantStack fit:** Already in use. No change needed.
- **Verdict: KEEP. Already bought.**

**pomegranate** -- PyTorch-based probabilistic models. HMMs, Bayesian networks, GMMs. GPU-accelerated.

- **QuantStack fit:** Only worth switching if you need GPU-accelerated regime detection or Bayesian network models. The PyTorch dependency is already in the RL extras.
- **Integration effort:** MEDIUM (3-5 days). Different API from hmmlearn.
- **Verdict: SKIP.** hmmlearn is sufficient. pomegranate adds GPU but regime detection is not the bottleneck.

---

### 6. Feature Engineering for ML

**Current state:** Custom feature engineering scattered across signal collectors and ML pipeline code.

| Package | Version | Stars | License | Install |
|---------|---------|-------|---------|---------|
| **tsfresh** | 0.21.1 | 9,168 | MIT | `pip install tsfresh` |
| **featuretools** | 1.31.0 | 7,627 | BSD-3 | `pip install featuretools` |
| **TA-Lib** | 0.6.8 | N/A | BSD-2 | `pip install TA-Lib` (requires C library) |
| **pandas-ta** | 0.4.71b | N/A | MIT-ish | `pip install pandas-ta` |

**tsfresh** -- Automated extraction of 794 time series features (statistical, spectral, entropy, etc.). Automatic relevance filtering via hypothesis testing.

- **QuantStack fit:** Could massively expand the feature space for ML models. The relevance filter (`select_features()`) is the killer feature -- it removes useless features automatically.
- **Gotcha:** Slow on large datasets without parallelization. Best run offline during research, not in hot path.
- **Integration effort:** MEDIUM (1 week). Need to format OHLCV data into tsfresh's expected long format. Run as research pipeline step.
- **Verdict: STRONG BUY for research pipeline.** Replaces manual "which lag/window/transform to try" exploration.

**featuretools** -- Deep Feature Synthesis across relational tables. Creates features from entity relationships.

- **QuantStack fit:** Less relevant for time series. Better for cross-table features (e.g., "average earnings surprise for stocks in same sector"). Could be useful for fundamental-based features.
- **Integration effort:** MEDIUM-HIGH (1-2 weeks). Need to define entity sets from DB tables.
- **Verdict: CONSIDER for fundamental/cross-asset feature engineering only.**

**TA-Lib** -- 150+ technical indicators (RSI, MACD, Bollinger, etc.) in C, with Python wrapper.

- **QuantStack fit:** Likely duplicates custom technical indicator code. Faster than pure Python implementations.
- **Gotcha:** Requires system-level C library install (`brew install ta-lib`). Docker needs extra build step.
- **Integration effort:** LOW (2-3 days) but Docker complexity.
- **Verdict: CONSIDER.** Only if current custom indicators are a maintenance burden.

**pandas-ta** -- Pure Python TA library, 130+ indicators, pandas DataFrame extension.

- **QuantStack fit:** Same as TA-Lib but no C dependency. Slightly slower but much easier to install.
- **Integration effort:** LOW (1-2 days). `df.ta.rsi()` etc.
- **Verdict: BUY as TA-Lib alternative if Docker simplicity matters.**

---

### 7. Transaction Cost Analysis (TCA)

**Current state:** No dedicated TCA module found.

No mature open-source Python TCA library exists. This is a gap in the ecosystem.

**Available options:**
- **empyrical-reloaded** -- basic cost-adjusted returns, but not true TCA
- **vectorbt** -- includes basic commission/slippage modeling in backtests
- **Custom** -- remains the only option for real TCA (implementation shortfall, VWAP slippage, market impact models)

**Verdict: BUILD.** No viable package exists. Build a lightweight TCA module (~200-400 LOC) that tracks:
- Implementation shortfall (decision price vs. fill price)
- Spread cost estimation
- Market impact (Almgren-Chriss or simpler square-root model)
- Feed results back to portfolio optimizer as cost constraints

---

### 8. Risk Management / VaR

**Current state:** `execution/risk_gate.py` (1,288 LOC). Risk gate is law per CLAUDE.md.

| Package | Version | Stars | License | Install |
|---------|---------|-------|---------|---------|
| **empyrical-reloaded** | 0.5.12 | ~1,478 | Apache-2.0 | `pip install empyrical-reloaded` |
| **quantstats** | 0.0.81 | 6,936 | Apache-2.0 | `pip install quantstats` |
| **Riskfolio-Lib** | 7.2.1 | 4,032 | BSD-3 | `pip install riskfolio-lib` |

**quantstats** -- Portfolio analytics: 200+ metrics, tear sheets, HTML reports. Sharpe, Sortino, VaR, CVaR, drawdown analysis, rolling metrics.

- **QuantStack fit:** Excellent for post-trade analysis and reporting. Could generate HTML tearsheets for every strategy.
- **Integration effort:** LOW (1-2 days). Feed returns series, get reports.
- **Verdict: STRONG BUY for analytics/reporting.** Do NOT use to replace risk_gate.py -- the risk gate must remain custom (it embodies QuantStack-specific invariants).

**Note:** Risk gate MUST remain custom code. It is the most critical 1,288 lines in the system. External packages can supplement (e.g., VaR calculation) but the gate logic, position limits, kill switch, and regime-conditional rules must be hand-written and hand-reviewed.

---

### 9. Database ORM vs Raw SQL

**Current state:** `db.py` (3,473 LOC). Raw SQL with psycopg3 + connection pooling.

| Package | Version | Stars | License | Install |
|---------|---------|-------|---------|---------|
| **SQLAlchemy 2.0** | 2.0.49 | N/A | MIT | `pip install sqlalchemy` |
| **SQLModel** | 0.0.38 | 17,798 | MIT | `pip install sqlmodel` |

**SQLAlchemy 2.0** -- The standard Python ORM. Full async support, type annotations, both ORM and Core modes. v2.0 is a major modernization.

- **QuantStack fit:** Could replace raw SQL in db.py. The 3,473 LOC of hand-written queries would become ~1,500 LOC of models + repository pattern.
- **Gotcha:** Migration from raw psycopg3 is a LARGE effort. db.py is deeply integrated. Any bugs in migration directly impact trading.
- **Integration effort:** HIGH (3-6 weeks). This is the highest-risk migration in the list.
- **Verdict: NOT NOW.** The raw SQL works, is tested, and is understood. The cost of migration (blast radius = every DB operation) far outweighs the benefit. Consider only if db.py maintenance becomes a bottleneck. If you do migrate, use expand-contract pattern: add SQLAlchemy models alongside existing queries, migrate one table at a time.

**SQLModel** -- Pydantic + SQLAlchemy fusion. Great for new projects, less suited for migration.

- **Verdict: SKIP.** Same concerns as SQLAlchemy, plus SQLModel is still pre-1.0 (v0.0.38).

---

### 10. Agent Orchestration / LangGraph

**Current state:** Already uses LangGraph (v0.4.11+ in deps). Three StateGraphs for research, trading, supervisor.

| Package | Version | Stars | License | Install |
|---------|---------|-------|---------|---------|
| **langgraph** | 1.1.6 | 28,631 | MIT | `pip install langgraph` |

**langgraph** -- Already in use. Current deps pin `>=0.4.11`, latest is 1.1.6.

- **Action:** UPGRADE to 1.x. Major improvements in 1.0: durable execution, better checkpointing, improved streaming, `langgraph-api` for deployment.
- **Breaking changes in 1.0:** State graph API largely compatible but some edge behaviors changed. Test thoroughly.
- **Integration effort:** MEDIUM (1 week). Version bump + regression testing.
- **Verdict: UPGRADE (high priority).** 0.4 to 1.1 is a significant maturity jump.

---

### 11. Data Providers

**Current state:** Alpha Vantage (primary, 75/min premium), Alpaca IEX (intraday), FRED, Edgar.

| Package | Version | Stars | License | Install |
|---------|---------|-------|---------|---------|
| **alpaca-py** | 0.43.2 | 1,223 | Apache-2.0 | `pip install alpaca-py` |
| **yfinance** | 1.2.0 | 22,751 | Apache-2.0 | `pip install yfinance` |
| **polygon-api-client** | 1.16.3 | 1,378 | MIT | `pip install polygon-api-client` |

**alpaca-py** -- Already a dependency. Current pin `>=0.20.0`, latest 0.43.2.

- **Action:** Upgrade. Significant improvements in data API, WebSocket stability, and order management.
- **Verdict: UPGRADE.**

**yfinance** -- Free, unlimited Yahoo Finance data. 22k stars. The most popular market data library.

- **QuantStack fit:** Good fallback/supplement for daily OHLCV when AV rate limits are exhausted. Free options chain data (delayed). Free fundamentals.
- **Gotcha:** Yahoo can break the API at any time (no official support). Not suitable as sole data source.
- **Integration effort:** LOW (1-2 days). Add as fallback provider.
- **Verdict: BUY as supplementary data source.** Reduces AV rate pressure.

**polygon-api-client** -- Polygon.io official client. Real-time and historical data. Paid plans.

- **QuantStack fit:** Best-in-class data quality. Real-time WebSocket, full options data, trades and quotes. Would be a significant upgrade over AV + Alpaca IEX.
- **Gotcha:** Paid ($29-199/mo depending on plan). Real-time requires higher tier.
- **Integration effort:** MEDIUM (3-5 days). New provider adapter.
- **Verdict: CONSIDER when revenue justifies the data cost.** Best data quality available.

---

### 12. Quant Analytics / General Purpose

| Package | Version | Stars | License | Install |
|---------|---------|-------|---------|---------|
| **quantstats** | 0.0.81 | 6,936 | Apache-2.0 | `pip install quantstats` |
| **empyrical-reloaded** | 0.5.12 | ~1,478 | Apache-2.0 | `pip install empyrical-reloaded` |

Both covered in sections above. **quantstats** for reporting, **empyrical-reloaded** for metric calculations.

---

## Summary: Recommended Actions

### Tier 1 -- High Value, Low Risk (do now)

| Package | Replaces | Effort | Why |
|---------|----------|--------|-----|
| **empyrical-reloaded** | Scattered metric calculations | 1-2 days | Standardizes Sharpe/Sortino/VaR. No reason to maintain custom. |
| **quantstats** | No existing reporting | 1-2 days | Instant tear sheets + HTML reports for every strategy. |
| **alphalens-reloaded** | Manual IC evaluation | 2-3 days | Rigorous factor evaluation. Quantile analysis, decay, turnover. |
| **yfinance** | AV rate limit pressure | 1-2 days | Free fallback data. Reduces AV dependency. |

### Tier 2 -- High Value, Medium Risk (next sprint)

| Package | Replaces | Effort | Why |
|---------|----------|--------|-----|
| **vectorbt** | Custom backtester (163 LOC) | 1-2 weeks | Current backtester is underbuilt. vectorbt adds walk-forward, Monte Carlo, optimization. |
| **PyPortfolioOpt** | Custom optimizer (902 LOC) | 3-5 days | HRP, Black-Litterman, risk parity out of box. Well-tested. |
| **tsfresh** | Manual feature engineering | 1 week | 794 auto-extracted features with relevance filtering. Research pipeline accelerator. |
| **langgraph 1.x** | langgraph 0.4.x | 1 week | Durable execution, better checkpointing. Major stability upgrade. |
| **alpaca-py** upgrade | alpaca-py 0.20 | 1-2 days | Current pin is 23 minor versions behind. |

### Tier 3 -- Conditional / Watch

| Package | Condition | Notes |
|---------|-----------|-------|
| **QuantLib** | When options pricing accuracy matters | Replace py_vollib. American exercise, vol surfaces. |
| **Riskfolio-Lib** | When needing CVaR/robust optimization | Superset of PyPortfolioOpt. More complex. |
| **skfolio** | After v1.0 | sklearn-compatible portfolio optimization. Promising but young. |
| **polygon-api-client** | When revenue justifies $29+/mo | Best data quality. Real-time options data. |
| **bt** | When portfolio-level backtesting needed | Complement to vectorbt. |

### Tier 4 -- Do Not Buy

| Package | Reason |
|---------|--------|
| **SQLAlchemy/SQLModel** | db.py migration blast radius too high. Raw SQL works fine. Revisit only if maintenance cost exceeds 20% of DB work. |
| **backtesting.py** | AGPL-3.0 license is viral. |
| **FinancePy** | GPL-3.0 license is viral. |
| **zipline-reloaded** | Heavy, opinionated, overkill. vectorbt is more pragmatic. |
| **pomegranate** | hmmlearn already works. No GPU bottleneck in regime detection. |

---

## Open Questions (Require Codebase Analysis)

1. **Two optimizer files exist** (`portfolio/optimizer.py` 308 LOC + `core/portfolio/optimizer.py` 594 LOC). Which is active? Consolidate before introducing PyPortfolioOpt.
2. **py_vollib status** -- not in pyproject.toml dependencies. Is it vendored, removed, or an unlisted dep? Determines urgency of QuantLib migration.
3. **Signal synthesis architecture** -- does `synthesis.py` (1,005 LOC) use weighted linear combination, or something more sophisticated (ensemble, stacking)? This determines whether alphalens can supplement or needs to co-exist.
4. **vectorbt Commons Clause** -- legal review needed. The clause prevents "selling the software" but not using it for trading. Should be fine for QuantStack's use case but confirm.
5. **langgraph 0.4 to 1.x breaking changes** -- need to audit all three graphs (research, trading, supervisor) for API compatibility before upgrade.
6. **Custom TCA** -- no package exists. Should be built as a new module `src/quantstack/analytics/tca.py`. Estimated 200-400 LOC. Feed into risk gate and portfolio optimizer.

---

## Dependency Budget Impact

Adding all Tier 1 + Tier 2 packages:

```
# New dependencies (Tier 1)
empyrical-reloaded>=0.5.12      # ~2MB, minimal deps
quantstats>=0.0.81               # ~5MB, depends on matplotlib
alphalens-reloaded>=0.4.6        # ~3MB, depends on seaborn
yfinance>=1.2.0                  # ~2MB

# New dependencies (Tier 2)
vectorbt>=0.28.5                 # ~15MB, depends on numba + plotly
pyportfolioopt>=1.6.0            # ~3MB, depends on cvxpy
tsfresh>=0.21.1                  # ~5MB, depends on statsmodels

# Upgrades
langgraph>=1.1.0                 # existing dep, version bump
alpaca-py>=0.43.0                # existing dep, version bump
```

Total new footprint: ~35MB of pure Python packages + their transitive deps. The heaviest is vectorbt (numba, plotly) but numba is likely already pulled in by other ML deps.
