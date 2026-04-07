# Interview Transcript — Phase 7: Feedback Loops & Learning

---

## Q1: Implementation Sequencing

**Q:** The spec lists 12 items with inter-dependencies (7.1 before 7.2, 7.2 before 7.6, Phase 2 dependency for IC-based items). Should we implement strictly in spec order, or can we parallelize independent streams?

**A:** Parallel streams where possible — group independent items and implement concurrently within each group.

---

## Q2: Ghost Module API Trust

**Q:** The 6 ghost modules have existing APIs but zero consumers. Should we trust and wire these existing APIs as-is, or should we audit/refactor the APIs before wiring (e.g., OutcomeTracker's affinity formula uses tanh(pnl/5.0) with 0.05 step — very slow to adapt)?

**A:** Audit and fix before wiring. Review each API's math/thresholds and fix obvious issues during integration.

---

## Q3: Phase 2 IC Tracking State

**Q:** Items 7.3, 7.7, 7.9, 7.10 all depend on Phase 2 item 2.1 (IC tracking). What's the current state of Phase 2?

**A:** IC tracking is in progress. User asked to check the codebase.

**Follow-up finding:** `signal_ic` table is populated nightly by `run_ic_computation()` (cross-sectional IC per strategy). But `ICAttributionTracker` (per-collector IC) and `SkillTracker.record_ic()` (per-agent IC) are dead code with zero data flow. The `signals` table being populated is the upstream dependency.

---

## Q4: Signal Weight Model

**Q:** For IC degradation (7.3) and signal correlation (7.7), should dynamic weights completely replace the static profiles, or should static weights serve as priors/floors?

**A:** Static as priors, IC adjusts. Keep regime profiles as baseline, multiply by IC-derived factors. Prevents a signal from getting zero weight just because IC dipped temporarily.

---

## Q5: Regime Transition Response

**Q:** For regime transition detection (7.12), how aggressive should the transition response be — especially given this is an autonomous system with no human override?

**A:** Moderate: reduce sizing 50%, don't block. Halve position sizes during transitions but still allow entries if conviction is high enough.

---

## Q6: EventBus

**Q:** Does an EventBus already exist in the system?

**A:** User asked to check the codebase.

**Finding:** Full EventBus exists at `src/quantstack/coordination/event_bus.py`. PostgreSQL-backed, poll-based, per-consumer cursors, 7-day TTL. Already has `IC_DECAY`, `DEGRADATION_DETECTED`, `MODEL_DEGRADATION`, `REGIME_CHANGE` event types. Just needs new types added (SIGNAL_DEGRADATION, SIGNAL_CONFLICT).

---

## Q7: EventBus Extension

**Q:** Any concerns about using the existing EventBus as the backbone for all Phase 7 feedback events?

**A:** Use it as-is, just add event types. The existing pattern is solid — extend with new types and wire publishers.

---

## Q8: Failure Mode Classifier Sophistication

**Q:** How sophisticated should the initial failure mode classifier be?

**A:** Hybrid: rules + LLM for ambiguous. Rule-based for obvious cases (regime mismatch — entry regime != exit regime), LLM for ambiguous classification.

---

## Q9: Model Registry Approach

**Q:** Should we build a lightweight custom registry or adopt MLflow?

**A:** Custom lightweight registry. DB table for metadata + file storage for model artifacts — simpler, no new dependency.

---

## Q10: Scheduling New Batch Jobs

**Q:** Should new scheduled jobs (loss aggregation, correlation analysis, drift detection) be added to supervisor batch or a separate service?

**A:** Add to supervisor batch. Keep all scheduled analysis in the supervisor graph — single orchestration point.

---

## Q11: Agent Degradation Alerts

**Q:** When an agent's win rate drops below 40%, what should the alert mechanism be?

**A:** Both: alert + auto-research. Publish AGENT_DEGRADATION EventBus alert immediately, then queue a lower-priority research task for prompt investigation.

---

## Q12: Conviction Calibration Factors

**Q:** Should the multiplicative conviction version preserve all 6 additive rules as factors, or simplify?

**A:** Convert all 6 to multiplicative. 1:1 mapping — each additive rule becomes a multiplicative factor.
