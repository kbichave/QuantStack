# Interview Transcript — Phase 9: Missing Roles & Scale

---

## Q1: Phase dependency status

**Q:** The spec lists dependencies on Phases 1-3 and Phase 5. Are EventBus wiring (1.7) and hardcoded model string removal (5.6) complete?

**A:** Prerequisites are done. Build on top of completed work.

---

## Q2: Alert lifecycle architecture

**Q:** Should Phase 9 add a SEPARATE system-level alert layer or EXTEND the existing equity alert tables?

**A:** Separate system-level alerts. New table + tools for operational/system alerts, keep equity alerts as-is.

---

## Q3: Discord notifications

**Q:** Should we use webhook-based or token-based Discord delivery?

**A:** Don't have Discord active. Add TODO note and don't work on it. Just integrate notifications to dashboard instead.

---

## Q4: Corporate actions data scope

**Q:** Should we include EDGAR 8-K parsing for M&A, or scope to AV dividends+splits only?

**A:** AV + EDGAR (full coverage). Include 8-K parsing for M&A detection.

---

## Q5: Extended hours enforcement

**Q:** How strict should "no new entries" in extended hours be — risk gate hard block or graph routing?

**A:** Risk gate enforcement (hard block). Risk gate rejects any order outside market hours. Safest — can't be bypassed by agent.

---

## Q6: Dashboard tech stack

**Q:** What's the dashboard tech stack?

**A:** Check the code, there is one. (Research found: Textual TUI with 6 tabs + FastAPI web dashboard on port 8421 with SSE.)

---

## Q7: Factor exposure configuration

**Q:** Should factor thresholds and benchmark be configurable?

**A:** Configurable everything. Configurable thresholds AND benchmark selection.

---

## Q8: Performance attribution placement

**Q:** Should per-cycle attribution run as a new trading graph node or as a tool?

**A:** New trading graph node (automatic). Runs every cycle after reflect — always-on attribution.

---

## Q9: EventBus ACK scope

**Q:** Which events should require ACK?

**A:** All risk events require ACK: RISK_WARNING, RISK_ENTRY_HALT, RISK_LIQUIDATION, RISK_EMERGENCY, IC_DECAY, REGIME_CHANGE, MODEL_DEGRADATION.

---

## Q10: Alert dashboard integration

**Q:** Should alerts appear in both dashboards (TUI + web) or just one?

**A:** Both dashboards. TUI gets an alerts widget on Overview; web dashboard gets an alerts pane or banner.
