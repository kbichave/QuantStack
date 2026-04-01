# Trading Rules Reference

Referenced by `trading_loop.md`. Lookup tables for sizing, holding periods, error handling, and alert monitoring. Decision logic (when to apply each) stays in the main loop.

---

## Position Sizing

**Equity:**

| Conviction | Size |
|------------|------|
| > 85% | Full (10% equity) |
| 70-85% | Half (5% equity) |
| 60-70% | Quarter (2.5% equity) |
| < 60% | SKIP |

**Options:**

| Conviction | Max Premium |
|------------|-------------|
| > 85% | 2% equity |
| 70-85% | 1.5% equity |
| 60-70% | 1% equity |
| < 60% | SKIP |

Always defer to risk desk recommendation if it differs from this table.

---

## Holding Period Guidelines

Positions are held as long as the thesis supports — could be hours, days, weeks, or months.

| Time Horizon | Typical Hold | Stop | Target | Trailing? | Exit Trigger |
|-------------|-------------|------|--------|-----------|-------------|
| Intraday | Same day | 1.0× ATR | 1.5× ATR | No | Market close |
| Swing | 3-10 days | 1.5× ATR | 2.5× ATR | Yes | Technical invalidation |
| Position | 1-8 weeks | 2.0× ATR | 3.0× ATR | Yes | Regime flip or thesis drift |
| Investment | 4-26 weeks | 2.5-3.0× ATR or fundamental floor | Fundamental fair value | 15-20% trailing | Thesis invalidation |

**Investment-grade exit criteria** (NOT just ATR-based):
- Piotroski F-Score drops below 5 (was >= 7 at entry)
- Two consecutive earnings misses
- Revenue deceleration for 2+ quarters
- Insider selling cluster (3+ insiders selling within 30 days)
- Valuation exceeds fair value estimate by > 20% (take profit)
- Macro regime shift invalidates sector thesis (e.g., rate hike kills growth thesis)

**Only intraday positions flatten at market close.** Swing, position, and investment trades carry overnight.

---

## Alert Monitoring Conditions

For each active/acted alert, check current price against these levels:

| Condition | Action |
|-----------|--------|
| Price ≤ stop_price | `create_exit_signal(alert_id, "stop_loss_hit", "critical", headline, exit_price=price, pnl_pct=pnl, commentary="...", recommended_action="close")` |
| Price ≥ target_price | `create_exit_signal(alert_id, "target_reached", "info", headline, exit_price=price, pnl_pct=pnl, recommended_action="close")` |
| Price dropped trailing_stop_pct% from high | `create_exit_signal(alert_id, "trailing_stop_hit", "critical", headline, ...)` |
| Regime changed from entry regime | `create_exit_signal(alert_id, "regime_flip", "warning", headline, what_changed="trending_up → ranging", ...)` |
| Held > max holding period | `create_exit_signal(alert_id, "time_stop", "warning", headline, recommended_action="close")` |

---

## Error Handling

| Failure | Response |
|---------|----------|
| `get_signal_brief` fails for a symbol | Skip that symbol for this iteration. Log warning. Do NOT trade on missing signals. |
| `execute_trade` returns error | Do NOT retry automatically. Log error + full params. Alert in trade_journal. |
| Risk desk agent unresponsive | Do NOT enter the trade. No sizing = no trade. |
| `get_portfolio_state` fails | STOP iteration. Cannot trade without knowing current positions. |
| Tool timeout | Skip that symbol, continue with others. "When in doubt, HOLD." |
| Multiple collectors down (>5) | Treat all signals as low quality. Monitor-only mode, no new entries. |
