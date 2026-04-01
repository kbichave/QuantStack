# Data Inventory & Transaction Costs

Referenced by `research_shared.md`. Use this as a lookup when scoping what data is available.

---

## Symbols

```python
# TARGET_SYMBOL narrows research to one stock. Unset = full watchlist.
_target = os.environ.get("TARGET_SYMBOL", "").upper()
if _target:
    symbols = [(s,) for s in _target.split(",")]
else:
    symbols = conn.execute("SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol").fetchall()
```

---

## Data Inventory

**Source: Alpha Vantage (premium, 75 calls/min).** Alpaca = paper execution only.

| Data | Coverage |
|------|----------|
| OHLCV | Daily/Weekly (~20yr). Intraday 5-min available if fetched via `acquire_historical_data.py --phases ohlcv_5min` |
| Options | 12K+ contracts/symbol, full Greeks (HISTORICAL_OPTIONS) |
| Fundamentals | Income stmt, balance sheet, cash flow, overview |
| Valuation | P/E, P/B, EV/EBITDA, FCF yield, dividend yield (from fundamentals) |
| Quality Factors | Piotroski F-Score, Novy-Marx GP, Sloan Accruals, Beneish M-Score |
| Growth Metrics | Revenue acceleration, operating leverage, earnings momentum (SUE) |
| Ownership | Insider cluster buys, institutional herding (LSV), analyst revision momentum |
| Earnings | History, estimates, call transcripts + LLM sentiment |
| Macro | CPI, Fed Funds, GDP, NFP, unemployment, treasury yield curve |
| Flow | Insider txns, institutional holdings, news sentiment |

---

## Transaction Cost Realism

**Flat 0.1% slippage for all instruments is unrealistic.** Use empirical costs when available,
fall back to defaults from `params["slippage_defaults"]`.

**Empirical-first approach:**
1. For any symbol with 20+ fills in the `fills` table, compute actual slippage from `get_tca_report(symbol)`. Use the empirical median as the backtest slippage assumption.
2. For symbols without fill history, use the default tier below.
3. Every 5 iterations, update `params["slippage_defaults"]` with medians from TCA data.

**Default slippage** (cold start / no fill history):

| Instrument | Default Slippage | Source |
|------------|-----------------|--------|
| Large-cap ETFs (SPY, QQQ) | `params["slippage_defaults"]["large_cap_etf"]` (0.05%) | Tight spreads, deep liquidity |
| Large-cap stocks | `params["slippage_defaults"]["large_cap_stock"]` (0.05%) | Sub-penny spreads |
| Mid-cap stocks | `params["slippage_defaults"]["mid_cap"]` (0.10%) | Wider spreads, less depth |
| Small-cap / low-volume | `params["slippage_defaults"]["small_cap"]` (0.15%) | Significant market impact |
| Options | bid-ask spread / 2 (or `params["slippage_defaults"]["options"]` if unknown) | Options have wide spreads |

**Cost sensitivity test (MANDATORY for promotion):** After any successful backtest, re-run at 2x your assumed slippage. If Sharpe drops below 0.5, the strategy is cost-fragile — it doesn't have enough edge to survive real execution costs. Document this in the strategy evaluation.
