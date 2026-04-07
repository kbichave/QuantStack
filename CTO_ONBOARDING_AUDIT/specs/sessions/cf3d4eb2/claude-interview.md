# Interview Transcript: Phase 6 Execution Layer

---

## Q1: Algo Scheduler Architecture

**Q:** The OMS already selects TWAP/VWAP based on ADV thresholds but never executes them. Should the new algo scheduler live inside `order_lifecycle.py` (extending the existing OMS) or as a separate `algo_scheduler.py` that the OMS delegates to after state transition to SUBMITTED?

**A:** Extend order_lifecycle.py — keep all order state management in one place.

---

## Q2: SEC Compliance Enforcement Model

**Q:** For SEC compliance (wash sale, PDT, tax lots): should these be pre-trade checks in the risk gate (blocking orders before submission) or post-trade accounting (recording and flagging after fills), or both?

**A:** Your call — architect it. (Decision: PDT is hard block in risk gate since account is below $25K. Wash sale tracking is post-trade accounting with pre-trade warning. Tax lots are post-trade accounting on every fill.)

---

## Q3: Fills Schema Migration

**Q:** The fills table uses order_id as PK (one row per order). For partial fill tracking (6.2), should we add a new `fill_legs` table alongside the existing `fills` table, or replace `fills` entirely with a legs-based schema?

**A:** Your call — architect it. (Decision: Add fill_legs alongside — backward compatible, existing code that reads fills doesn't break.)

---

## Q4: TCA Feedback Approach

**Q:** The Almgren-Chriss module already has calibration-from-fills code (needs ≥10 fills). For the TCA EWMA feedback loop (6.1), should we wire into that existing calibration or build a simpler EWMA that runs after every fill?

**A:** New EWMA layer on top — simpler EWMA that updates after every fill, converges faster, uses 2x multiplier early until 50 fills.

---

## Q5: Paper Broker Volume Data

**Q:** For the paper broker TWAP/VWAP simulation — do you have historical 1-minute or 5-minute bar data available for volume profiles, or should the paper broker use synthetic U-shaped volume curves?

**A:** Have intraday bar data — historical 1-min or 5-min bars available for realistic volume profiles.

---

## Q6: NBBO Source for Audit Trail

**Q:** For the best execution audit trail (6.6) — where should NBBO snapshots come from?

**A:** Alpaca IEX quotes (15-min delayed) — free, already integrated. Acceptable for paper trading audit.

---

## Q7: Options Exit Behavior

**Q:** For options monitoring rules (6.7) — when theta acceleration, pin risk, IV crush, assignment risk trigger, should the system auto-exit, flag, or roll?

**A:** Configurable per rule — some rules auto-exit (e.g., pin risk), some flag (e.g., IV crush). Per-rule configuration.

---

## Q8: Account Size and PDT

**Q:** What's the current account equity — above or below $25K PDT threshold?

**A:** Below $25K — PDT enforcement is critical path. Must enforce to avoid regulatory violation.

---

## Q9: Short Positions and Borrow Costs

**Q:** Are you currently holding or planning to hold short equity positions?

**A:** Options only (no equity shorts) — funding cost is simpler, mainly margin interest on leveraged longs.

---

## Q10: Scope Priority

**Q:** The spec has 9 items at 23-27 days. What's the priority if we need to cut scope?

**A:** Ship all 9 — no cuts. Full scope, accept the timeline.
