# P15 Self-Interview: Autonomous Fund Integration

## Q1: What does "autonomous" actually mean — what decisions can the system make without human input?
**A:** Autonomous means: enter/exit positions (within risk limits), promote/demote strategies (via data-driven gates), adjust signal weights (via IC attribution), prioritize research (via scoring), and improve agent prompts (via A/B testing). It does NOT mean: exceed risk ceilings, deploy untested models, or modify its own core logic. Every autonomous decision has a ceiling and a circuit breaker.

## Q2: How do you verify that feedback loops are actually closing?
**A:** Each loop has a measurable trigger and a measurable behavior change. Example — Loop 3 (IC degradation → signal weight): trigger = collector IC drops below 0.02, expected behavior = collector weight decreases within 24h. The loop verifier checks: did the trigger fire? Did the weight change? If trigger fired but weight didn't change → loop is broken → alert.

## Q3: What happens during the 7-day burn-in if something goes wrong?
**A:** The burn-in runs with all safety layers active (risk gate, kill switch, circuit breakers). If a bug triggers a kill switch → burn-in fails, fix the bug, restart the 7-day clock. If a REAL risk trigger fires (drawdown from legitimate market move) → this is expected behavior, burn-in continues. The distinction: infrastructure failures reset the clock, market-driven safety triggers do not.

## Q4: How does position reconciliation handle corporate actions (splits, dividends)?
**A:** Corporate actions cause legitimate position differences between system and broker. The reconciler checks for corporate action events (via data provider) before flagging mismatches. If a 2:1 split occurred and system shows 100 shares but broker shows 200 → adjust system state, no alert. If no corporate action and mismatch exists → alert + log.

## Q5: What's in the weekly automated report?
**A:** Five sections: (a) Performance: weekly Sharpe, drawdown, Calmar, Sortino vs benchmarks; (b) Attribution: signal alpha, execution alpha, timing alpha; (c) Winners/Losers: top 3 each with causal attribution (P13); (d) Research: hypotheses generated/validated/rejected, strategy lifecycle events; (e) Health: uptime, error rate, loop health, data staleness. Output: markdown file + optional Discord summary.

## Q6: How do operating modes handle edge cases (holidays, early close, market halt)?
**A:** Market calendar integration (pandas_market_calendars). Holidays → overnight mode all day. Early close (day before Thanksgiving) → market mode ends at 13:00 ET instead of 16:00. Market halt (circuit breaker) → detected via data feed, switch to monitoring-only mode, resume when halt lifts. Mode transitions are event-driven, not purely clock-driven.

## Q7: What's the disaster recovery RTO/RPO?
**A:** RPO (Recovery Point Objective): 24h — daily pg_dump means max 24h of data loss. RTO (Recovery Time Objective): 30 min — Docker restart from last backup. Acceptable because: (a) positions are reconciled with broker (broker state survives), (b) ML models are checkpointed to disk (survive container restart), (c) only research queue and recent IC observations might be lost.

## Q8: How do you prevent the system from degrading slowly without anyone noticing?
**A:** Multiple degradation detectors: (a) loop health — if any feedback loop hasn't closed in 48h, alert; (b) IC trend monitoring — if overall portfolio IC trends down for 2 weeks, alert; (c) agent quality drift — if average agent quality drops below 45%, alert; (d) weekly report — automated Sharpe/drawdown tracking surfaces slow degradation visually. The supervisor graph runs these checks every cycle.
