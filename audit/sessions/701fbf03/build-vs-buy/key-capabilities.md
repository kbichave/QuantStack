# Build-vs-Buy Analysis: Key Capabilities

## 1. Hyperparameter Optimization

**Need:** Replace hardcoded ML hyperparameters with Bayesian optimization. Ref: P03, QS-M1.

| Option | Install | Stars | Last Release | License |
|--------|---------|-------|-------------|---------|
| **Optuna** | `pip install optuna==4.2` | 11.5K | 2026-03 | MIT |
| Hyperopt | `pip install hyperopt==0.2.7` | 7.3K | 2024-11 | BSD |
| Ray Tune | `pip install ray[tune]` | 35K | 2026-03 | Apache 2.0 |

**Recommendation: Optuna** — best docs, native LightGBM/XGBoost integration, supports purged CV via custom objectives, lightweight. Ray Tune is overkill for single-machine.

---

## 2. Portfolio Optimization

**Need:** HRP, risk parity, Kelly. Ref: P02, core/portfolio/.

| Option | Install | Stars | License |
|--------|---------|-------|---------|
| **PyPortfolioOpt** | `pip install pyportfolioopt==1.5.6` | 4.5K | MIT |
| Riskfolio-Lib | `pip install riskfolio-lib==6.4` | 2.8K | BSD |
| skfolio | `pip install skfolio==0.5` | 900 | BSD |

**Recommendation: Riskfolio-Lib** — most comprehensive (HRP, risk parity, CVaR, worst-case, 30+ methods). PyPortfolioOpt is simpler but less flexible. skfolio is scikit-learn compatible but newer.

---

## 3. Options Pricing (Advanced)

**Need:** Vol surface, exotic pricing. Ref: P06, P08.

| Option | Install | Stars | License |
|--------|---------|-------|---------|
| **QuantLib-Python** | `pip install QuantLib==1.35` | 5K (C++ core) | BSD |
| py_vollib (existing) | Already installed | 600 | MIT |
| FinancePy (existing) | Already installed | 2K | GPL-3.0 |

**Recommendation: QuantLib-Python** — industry standard for vol surface fitting, exotic pricing, Greeks computation. py_vollib is fine for basic Black-Scholes (keep for speed). FinancePy good for structured products.

---

## 4. Causal Inference

**Need:** Causal alpha discovery. Ref: P13.

| Option | Install | Stars | License |
|--------|---------|-------|---------|
| **DoWhy** | `pip install dowhy==0.11` | 7K | MIT |
| CausalML | `pip install causalml==0.16` | 5K | Apache 2.0 |
| EconML | `pip install econml==0.15` | 3.8K | MIT |

**Recommendation: All three** — DoWhy for causal graph discovery, CausalML for treatment effects, EconML for double ML. They're complementary, not competitive.

---

## 5. Conformal Prediction

**Need:** Calibrated prediction intervals. Ref: P14.

| Option | Install | Stars | License |
|--------|---------|-------|---------|
| **MAPIE** | `pip install mapie==0.9` | 1.3K | BSD |
| conformalprediction | `pip install conformalprediction` | 200 | MIT |

**Recommendation: MAPIE** — scikit-learn compatible, well-documented, supports all major conformal methods. Wraps existing LightGBM/XGBoost directly.

---

## 6. Transformer Forecasting

**Need:** Time series transformers. Ref: P14.

| Option | Install | Stars | License |
|--------|---------|-------|---------|
| **NeuralForecast** | `pip install neuralforecast==1.8` | 3K | Apache 2.0 |
| Chronos | `pip install chronos-forecasting` | 2.5K | Apache 2.0 |
| Lag-Llama | GitHub only | 1K | Apache 2.0 |

**Recommendation: NeuralForecast** — includes PatchTST, iTransformer, TFT. One package, many architectures. Chronos is interesting (foundation model) but less customizable.

---

## 7. Backtesting Framework

**Need:** Replace/augment custom walk-forward engine. Ref: P04.

| Option | Install | Stars | License |
|--------|---------|-------|---------|
| vectorbt | `pip install vectorbt==0.26` | 4.5K | Apache 2.0 |
| backtesting.py | `pip install backtesting==0.3.3` | 5.5K | AGPL-3.0 |
| Custom (existing) | Already built | N/A | Internal |

**Recommendation: Keep custom** — the existing walk-forward engine is well-designed. vectorbt is fast for vectorized backtests but doesn't replace the multi-agent execution simulation QuantStack needs. Use vectorbt for initial screening (fast) → custom for final validation (accurate).

---

## 8. Database ORM

**Need:** Replace 140k LOC raw SQL monolith. Ref: P07.

| Option | Install | Stars | License |
|--------|---------|-------|---------|
| SQLAlchemy 2.0 | `pip install sqlalchemy==2.0` | 9.5K | MIT |
| SQLModel | `pip install sqlmodel==0.0.22` | 14K | MIT |
| Raw SQL (existing) | Already built | N/A | Internal |

**Recommendation: Incremental extraction, NOT full ORM migration.** Migrating 140k LOC to an ORM is a multi-month rewrite with high regression risk. Instead: extract `db.py` into `db/schema.py`, `db/queries.py`, `db/migrations.py`. Add SQLAlchemy for NEW tables only. Keep existing raw SQL working.

---

## 9. RL Framework

**Need:** Production RL for trading. Ref: P09.

| Option | Install | Stars | License |
|--------|---------|-------|---------|
| **FinRL** (existing) | Already installed | 10K | MIT |
| ElegantRL | `pip install elegantrl==0.3.8` | 3.5K | Apache 2.0 |
| RLlib | `pip install ray[rllib]` | 35K (Ray) | Apache 2.0 |
| CleanRL | `pip install cleanrl==1.2` | 5K | MIT |

**Recommendation: Start with FinRL** (already integrated), evaluate ElegantRL for GPU-optimized training if FinRL is too slow. RLlib is production-grade but overkill for single-machine.

---

## 10. Alternative Data

**Need:** Non-traditional data sources. Ref: P11.

| Source | Cost | API | Priority |
|--------|------|-----|----------|
| **Quiver Quantitative** | Free tier | REST | HIGH — congressional trades, lobbying |
| SimilarWeb | $100+/mo | REST | MEDIUM — web traffic |
| Thinknum | $500+/mo | REST | DEFER — expensive for <$100K |
| USPTO | Free | REST | LOW — patent filings |
| MarineTraffic | $200+/mo | REST | DEFER — shipping data |

**Recommendation:** Start with Quiver (free congressional trades) and USPTO (free patents). Add SimilarWeb if budget allows. Defer expensive sources until portfolio justifies the data cost.
