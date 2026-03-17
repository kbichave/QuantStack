---
name: invest
description: Long-term fundamental investing session — weekly cadence, DCF-informed valuation, quality scoring, and portfolio concentration checks. For equity positions held weeks to months.
user_invocable: true
---

# /invest — Long-Term Fundamental Investing Session

## Purpose

Evaluate long-term equity investment opportunities using fundamentals, valuation,
and quality scoring. This is a **weekly-cadence** skill — not a daily trading workflow.
Decisions are held for weeks to months; exits are driven by fundamental deterioration
or price target, not ATR stops.

This skill does NOT use `run_analysis` or `get_signal_brief`. Purely fundamentals.

---

## Workflow

### Step 0: Read Context

- Read `.claude/memory/strategy_registry.md` — are there active investment strategies?
- Read `.claude/memory/trade_journal.md` — any open investment positions already?
- Read `.claude/memory/session_handoffs.md` — pending fundamental flags from prior sessions

### Step 1: Regime Gate

Call `get_regime(symbol)` for each candidate.

| Regime | Action |
|--------|--------|
| `trending_up` | Full sizing eligible |
| `ranging` | Full sizing eligible |
| `trending_down` | Reduce exposure — skip new entries, review existing |
| `unknown` | Paper mode only |

Investment positions are longer-term and less regime-sensitive than swing trades,
but a persistent `trending_down` regime with falling fundamentals warrants caution.

### Step 2: Screen (if no symbols provided)

Call `screen_stocks` to generate candidates. Minimum filters:
```
market_cap_gt=1_000_000_000   (>$1B — institutional-grade liquidity)
fcf_yield_gt=3                (>3% FCF yield — not a speculative growth name)
debt_to_equity_lt=1.5         (below 1.5x leverage)
roe_gt=12                     (>12% return on equity — quality floor)
```

Alternatively: user provides a specific symbol or list (skip this step).

### Step 3: Fundamental Scorecard

For each candidate, call these **in parallel**:
- `get_financial_metrics(symbol)` → P/E, P/FCF, EV/EBITDA, ROE, FCF yield, debt/equity
- `get_earnings_data(symbol)` → EPS trend, surprise history, next earnings date
- `get_insider_trades(symbol, days=90)` → net buy/sell directional signal
- `get_analyst_estimates(symbol)` → consensus direction, revision trend (up/down/flat)

Score each dimension on a 0–10 scale:

| Dimension | What to assess | 0-3 | 4-6 | 7-10 |
|-----------|---------------|-----|-----|------|
| **Quality** | ROE, FCF yield, debt/equity | Weak (ROE<10, high debt) | Average | Strong (ROE>18, FCF>5%) |
| **Value** | P/FCF, EV/EBITDA | Expensive (P/FCF>30) | Fair (15–30) | Cheap (<15) |
| **Momentum** | EPS trend, surprise history | Misses, declining | Flat | Beats, accelerating |
| **Insider Signal** | Net 90-day buy/sell | Net selling | Neutral | Net buying |

**Composite score** = (Quality × 0.35) + (Value × 0.30) + (Momentum × 0.20) + (Insider × 0.15)

### Step 4: Simple Valuation (DCF Shortcut)

For candidates scoring ≥ 6 composite:

```
fair_value = current_fcf_per_share × (1 + growth_est)^5 / (discount_rate - terminal_growth)

Where:
  current_fcf_per_share = from get_financial_metrics
  growth_est            = conservative (use 5-year analyst estimate, cap at 15%)
  discount_rate         = 0.10 (10% WACC — conservative for US equities)
  terminal_growth       = 0.03 (long-run GDP proxy)
```

**Margin of safety** = (fair_value - current_price) / fair_value

Require ≥ 20% margin of safety before entry.
Document your growth_est assumption and why it's conservative.

> This is judgment-based arithmetic, not a precision model. The goal is to identify
> whether a stock is roughly cheap (>30% MOS), fairly priced (10–20% MOS), or expensive.
> A 20% MOS means the market has to be 20% wrong before you lose money at this price.

### Step 5: Portfolio Concentration Check

Call `get_portfolio_state()`:
- Check sector concentration: no single sector > 30% of equity
- Check existing positions in the same industry (not just sector)
- Check total equity exposure to long-term positions vs swing positions
- If adding this position would breach a limit: reduce size to the limit, or skip

### Step 6: Decision

**Conviction tiers** (separate from equity swing sizing):

| Tier | Criteria | Position size |
|------|----------|---------------|
| **High** | Score ≥ 8, MOS ≥ 30%, no earnings within 14d | 5% of equity |
| **Moderate** | Score 6–8, MOS ≥ 20% | 2.5% of equity |
| **Low** | Score 4–6, MOS ≥ 20% | 1.25% of equity |
| **SKIP** | Score < 4, OR MOS < 20%, OR earnings within 7d | Do not enter |

**Earnings risk rule:** If earnings within 14 days, reduce size by 50% or skip.
Fundamental investing is about owning the business, not guessing the print.

### Step 7: Register Strategy + Execute

If this is a new long-term investment approach, register it first (once):
```python
register_strategy(
    name="fundamental_quality_invest",
    entry_rules=[],          # fundamental, not rule-based
    exit_rules=[],
    parameters={"min_mos": 0.20, "min_composite_score": 6.0},
    description="Long-term equity: FCF yield + quality + MOS ≥ 20%",
    time_horizon="investment",
    holding_period_days=180,
    instrument_type="equity",
    regime_affinity=["trending_up", "ranging"],
)
```

Then execute:
```python
execute_trade(
    symbol=symbol,
    action="buy",
    reasoning=f"Composite score {score:.1f}/10, MOS {mos:.0%}, quality metrics: {key_metrics}",
    confidence=confidence,          # 0.7 for Low, 0.8 for Moderate, 0.9 for High
    position_size=position_size,    # from conviction tier above
    strategy_id=strategy_id,
    paper_mode=True,
)
```

### Step 8: Update Memory

`trade_journal.md` entry must include:
- Fundamental scorecard table (all 4 dimensions with scores)
- DCF fair value and margin of safety
- Conviction tier and position size rationale
- Next review trigger: earnings date, price target, or weekly fundamental check

`strategy_registry.md`: confirm strategy is registered with `instrument_type="equity"`, `time_horizon="investment"`.

---

## Exit Criteria (use in /review)

Long-term positions are NOT stopped out by ATR. Exit when:
1. **Fundamental deterioration**: FCF yield drops below 2% OR ROE drops below 8% → exit
2. **Price target reached**: price exceeds fair_value × 1.1 (10% above fair value) → take profits
3. **Thesis broken**: earnings miss 2 consecutive quarters with guidance cut → exit
4. **Regime persists down**: `trending_down` for 4+ weeks AND fundamentals weakening → exit
5. **Time stop**: if position has not achieved 10% return in 12 months → review for exit

---

## What This Skill Does NOT Do

- No `run_analysis` or `get_signal_brief` — no technical IC crew
- No ATR-based stop loss (exit by fundamental review)
- No intraday or daily price-action decisions — weekly cadence only
- No speculative growth names without FCF (pre-profit companies → skip)
- No more than weekly review of investment positions (reviewing daily = noise)
