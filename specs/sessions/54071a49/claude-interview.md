# Stakeholder Interview — QuantStack 24/7 Readiness

**Date:** 2026-04-07
**Interviewer:** Deep-Plan Architect
**Interviewee:** Kshitij Bichave (Principal Quant Researcher)

---

## Q1: What's your target capital allocation for the first month of live trading?

**Answer:** $5K-$10K (conservative)

Small positions, tight stops, max 5 concurrent positions. Goal is to validate the system before scaling.

---

## Q2: When the system encounters a situation it hasn't seen before (flash crash, circuit breaker halt, earnings surprise outside model range), what should the default behavior be?

**Answer:** Defensive exit + alert

Close all positions at market, trigger kill switch, send alert. Prioritizes capital preservation over staying in the market.

---

## Q3: How should overnight/weekend research prioritize its compute budget?

**Answer:** Balanced (70/30 new/refine)

70% of compute budget on new hypothesis exploration, 30% on refining and improving existing winning strategies. Appropriate for early stage where strategy coverage is still growing.

---

## Q4: What alerting channels do you want for critical events?

**Answer:** Email only (Gmail SMTP)

Discord is not set up. Use Gmail with app password for all alerts. Simple and sufficient for personal monitoring.

---

## Q5: PostgresSaver migration strategy — parallel run or direct cutover?

**Answer:** Direct cutover (faster)

Since this is paper trading, risk of checkpoint corruption is low. If checkpoints corrupt, just restart the graph cycle. No need for the complexity of parallel validation.

---

## Q6: Which phases are must-haves before putting real money in?

**Answer:** Phase 1 + 2 + 3 (full autonomy) — all three phases required before live trading

Most conservative approach. Wants multi-mode operation, overnight research, loss analysis, budget tracking, and all safety/resilience features in place before deploying real capital. ~8 weeks of development.

---

## Q7: Email service preference for alerting?

**Answer:** Gmail SMTP

Simplest setup. Use a Gmail account with app password. Good enough for personal alerts at this scale.

---

## Q8: Intraday circuit breaker thresholds — based on daily P&L or total portfolio value?

**Answer:** Both (layered)

Daily P&L thresholds for intraday protection (resets each morning) PLUS portfolio-level thresholds for multi-day drawdown tracking (high-water mark based). Most comprehensive protection.

---

## Q9: What to do with 92 stubbed tools?

**Answer:** Prioritize top 10-15 stubs for implementation

Identify the most impactful stubs, implement them in Phase 2-3. Remove the rest from agent bindings so agents don't waste LLM calls on non-functional tools.

---

## Q10: Groq/Llama structured output fallback strategy?

**Answer:** Test first, decide later

Run a structured output benchmark on Groq during Phase 1 implementation. Make the Haiku-vs-Groq decision based on actual error rates rather than assumptions. Pragmatic data-driven approach.

---

## Key Design Decisions Summary

| Decision | Choice | Impact |
|----------|--------|--------|
| Capital | $5-10K conservative | Max 5 positions, tight stops |
| Unknown state behavior | Defensive exit + alert | Kill switch on unknowns |
| Research priority | 70% new / 30% refine | Balanced exploration |
| Alerting | Email only (Gmail SMTP) | No Discord dependency |
| PostgresSaver migration | Direct cutover | No parallel validation period |
| Go-live gate | Phase 1+2+3 complete | ~8 weeks before real money |
| Circuit breaker | Layered (daily + portfolio) | Most comprehensive |
| Stubbed tools | Prioritize top 10-15 | Implement best, hide rest |
| Groq structured output | Benchmark first | Data-driven provider choice |
