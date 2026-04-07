# Interview Transcript — Phase 4: Agent Architecture & Coordination

## Q1: Race condition conflict resolution location

**Question:** The spec says exits take priority at merge_parallel. Research shows merge_parallel is a no-op and state uses `Annotated[list, operator.add]` (accumulates, no reducer conflict). The real conflict is semantic — same symbol in both exit and entry lists. Where should resolution happen: (a) merge_parallel node, (b) post-merge validation step, (c) risk_sizing?

**Answer:** Post-merge validation step. Add a new node between merge_parallel and risk_sizing that validates cross-branch consistency.

---

## Q2: Pydantic state migration scope

**Question:** 3 TypedDict state classes exist (TradingState, ResearchState, SupervisorState). Migrate all at once or incrementally?

**Answer:** All 3 graphs at once. One migration, consistent codebase. No half-migrated state.

---

## Q3: Circuit breaker design pattern

**Question:** Strategy-level breaker (strategy_breaker.py) already exists. How should node-level breaker be designed?

**Answer:** Decorator pattern on node functions. `@circuit_breaker(threshold=3)` on each node. State stored in DB.

---

## Q4: DLQ alert routing

**Question:** Should DLQ alerts feed into the self-healing loop, be Langfuse-only, or both?

**Answer:** **Langfuse only, but strengthened.** Critical pushback on self-healing integration:

- The self-healing loop is not safe for prompt auto-patching in a live trading system. It was designed for tool errors (deterministic failures with known fix patterns). Prompt degradation has multiple possible causes (regime change, data drift, bad deployment, genuine prompt rot).
- **Architectural principle:** automate remediation only when the cause is unambiguous and the fix is reversible. Tool errors qualify. Prompt changes do not.
- DLQ rate tracked as Langfuse metric with 24h rolling window
- Two thresholds: warn at 5%, critical at 10%
- Critical alert fires outbound notification (Slack webhook or email) — don't let it sit silently
- Alert payload: which agent/node, sample of failed messages, rate trend (rising vs flat)
- DLQ metrics also surfaced on the system dashboard (alongside existing health metrics)
- Human reviews and decides if it's a prompt issue, deploys fix manually
- **Revisit after 60+ days of paper trading data** with clear correlation patterns

---

## Q5: Priority assignment mechanism

**Question:** Should message priorities be static (agents.yaml), dynamic (content-based), or hybrid?

**Answer:** Hybrid — config defaults + type overrides. agents.yaml sets agent default priority tier. Risk/execution messages always P0 regardless of source agent.

---

## Q6: Risk gate modification approach

**Question:** Risk gate is "LAW" in CLAUDE.md ("Never bypass. Never modify."). How to add pre-trade correlation, daily heat, sector concentration?

**Answer:** **Add directly to risk_gate.check(), but update CLAUDE.md first.**

- The "never modify" rule exists to prevent weakening. Adding stronger checks is the opposite action — in the spirit of the rule.
- Option 2 (pre-gate layer) is dangerous: two enforcement layers, harder to reason about, independently bypassable.
- Option 3 (subclass) creates silent-skip risk if any code path instantiates base class instead of subclass.
- **Process:**
  1. Update CLAUDE.md: change "Never modify" to "Never weaken or bypass"
  2. Add new checks inside risk_gate.check() with unit tests
  3. New checks fail closed — if correlation data unavailable, check blocks, does not pass
- "Ambiguous hard rules are more dangerous than no rules. People route around them."

---

## Q7: Node blocking classification

**Question:** Which of the 16 nodes should be classified as blocking (failure halts pipeline)?

**Answer:** Custom classification based on principle: **"If this node fails and pipeline continues, can we lose money unintentionally?"** Yes = blocking.

**Blocking (feeds risk calculations or manages existing exposure):**
- `data_refresh` — stale data means all downstream decisions wrong
- `safety_check` — non-negotiable gate
- `position_review` — failure means risk_sizing doesn't know current exposure, can breach limits
- `risk_sizing` — can't execute without valid sizes
- `execute_exits` — failing to close leaves unintended exposure open

**Non-blocking (generates new opportunities, safe defaults exist):**
- `plan_day` — defaults to neutral bias, no trades
- `entry_scan` — defaults to empty candidate list
- `portfolio_construction` — empty universe is safe
- `execute_entries` — no new positions, annoying not dangerous
- `resolve_symbol_conflicts` — safe default is drop all conflicted entries

**Pattern:** "Getting into trades is optional. Getting out of bad trades and knowing your current exposure is not."

---

## Q8: Regime-at-entry storage

**Question:** Where to store regime-at-entry: MonitoredPosition, DB, or both?

**Answer:** Both DB + MonitoredPosition. DB is source of truth, MonitoredPosition is runtime cache. Reconstruct on restart.

---

## Q9: Tool access control violation handling

**Question:** Should violations hard-reject, circuit-break, or log-only?

**Answer:** Hard reject + security event log. Block the call, return error to agent, log as security event. No circuit-breaking on violations.

---

## Q10: Event bus database

**Question:** Is event bus using PostgreSQL or SQLite?

**Answer:** PostgreSQL (same as rest of system). Use `INSERT ... ON CONFLICT (consumer_id) DO UPDATE SET ...`

---

## Q11: Deployment strategy

**Question:** Ship incrementally, batch by dependency, or atomic release?

**Answer:** Atomic release. All Phase 4 items deploy together.

---

## Q12: Pydantic node return types

**Question:** Should nodes continue returning dicts, use typed output models, or partial state updates?

**Answer:** Typed output models per node. Each node gets a NodeOutput Pydantic model. Maximum type safety.
