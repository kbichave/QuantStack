# Phase 10: Advanced Research — Deep Plan Spec

**Timeline:** Week 10+ (4 sub-phases over ~20 weeks)
**Effort:** ~100 days (50 days with 2 engineers)
**Gate:** Phases 1-9 substantially complete. System profitable in paper trading with validated signals.

---

## Context

This spec is part of the QuantStack CTO Onboarding Audit implementation plan (164 findings, overall grade C-). Phase 10 is the visionary phase — autoresearch loops, metacognitive self-modification, knowledge graphs, hierarchical governance. These are the right destination, but building them on unvalidated signals and phantom execution would be building a self-improving machine that improves in the wrong direction.

**Full audit reference:** [`CTO_ONBOARDING_AUDIT/`](../README.md)
**Primary audit section:** [`10_ADVANCED_RESEARCH.md`](../10_ADVANCED_RESEARCH.md)
**Prerequisites:** Sections 01-06 substantially complete before starting.

---

## Objective

Transform QuantStack from a trading system into a self-improving trading company. The system doesn't just make money — it gets better at making money, and gets better at getting better.

---

## Sub-Phase 10A: Foundation (Week 10-11, ~3 weeks)

### AR-8: Dynamic Tool Lifecycle (Mimosa Pattern)

- **Effort:** 1 week
- **Audit section:** [`10_ADVANCED_RESEARCH.md` §AR-8](../10_ADVANCED_RESEARCH.md)
- **Problem:** 92 of 122 tools are stubs. Agents offered tools that return errors, wasting LLM round-trips.
- **Fix:**
  1. **Registry Cleanup:** Split `TOOL_REGISTRY` into `ACTIVE_TOOLS` (~30) and `PLANNED_TOOLS` (~92 stubs). Agents only see active.
  2. **Health Monitoring:** Track invocation count, success rate, latency per tool. Disable <50% success rate.
  3. **Tool Synthesis:** AutoResearchClaw implements highest-priority planned tools based on demand signals.
  4. **Capability Announcement:** `TOOL_ADDED` event → agents reload bindings next cycle.
- **Acceptance criteria:**
  - [ ] Agents only see active, working tools
  - [ ] Tool health monitoring active
  - [ ] AutoResearchClaw implements tools based on demand

### AR-7: Error-Driven Research Iteration (MAGNET + ResearchGym)

- **Effort:** 1 week
- **Depends on:** Phase 7 (failure mode taxonomy)
- **Audit section:** [`10_ADVANCED_RESEARCH.md` §AR-7](../10_ADVANCED_RESEARCH.md)
- **Problem:** No systematic connection from losses to research priorities based on failure mode analysis.
- **Fix:** Daily loss analysis pipeline (16:30 ET): COLLECT losses → CLASSIFY by failure mode → AGGREGATE over 30 days → PRIORITIZE by P&L impact → GENERATE research tasks → FEED research_queue.
- **Acceptance criteria:**
  - [ ] Daily loss analysis pipeline runs automatically
  - [ ] Research tasks generated targeting top failure modes
  - [ ] Closed loop: loss → research → new strategy/hedge → prevented future loss

### AR-9: Fixed-Budget Experiment Discipline (ResearchGym)

- **Effort:** 1 week
- **Audit section:** [`10_ADVANCED_RESEARCH.md` §AR-9](../10_ADVANCED_RESEARCH.md)
- **Problem:** No experiment budgets. Runaway research agents can consume unlimited tokens.
- **Fix:** Per-cycle budgets: 50K tokens (research), 30K (trading). Per-cycle cost: $0.50 (research), $0.20 (trading). Prioritization: `priority = (expected_IC * regime_fit * novelty_score) / estimated_compute_cost`. Patience protocol: never discard after 1 backtest — run 3 time windows.
- **Acceptance criteria:**
  - [ ] Per-cycle token and cost budgets enforced
  - [ ] Experiment prioritization formula implemented
  - [ ] 3-window patience protocol prevents premature rejection

---

## Sub-Phase 10B: Research Multiplier (Week 11-15, ~5 weeks)

### AR-1: Overnight Autoresearch Loop (Karpathy Pattern)

- **Effort:** 1 week | **Impact:** 16x research throughput
- **Audit section:** [`10_ADVANCED_RESEARCH.md` §AR-1](../10_ADVANCED_RESEARCH.md)
- **Concept:** Fixed 5-minute experiment budgets. ~96 experiments per night (20:00-04:00 ET). Single metric: OOS IC. Haiku for hypothesis generation. Discard failures immediately. Winners → morning validation.
- **Fix:**
  1. Add `overnight_mode` to research graph runner
  2. Create `autoresearch_node.py` with fixed 5-min budget
  3. Haiku for hypothesis generation (speed > depth)
  4. Single metric: OOS IC on purged holdout
  5. Store log in `autoresearch_experiments` table
  6. Morning pipeline evaluates winners with full backtest
- **Acceptance criteria:**
  - [ ] Overnight mode runs 8 hours producing ~96 experiments
  - [ ] Winners automatically queued for morning full validation
  - [ ] Experiment log queryable for research quality tracking

### AR-10: Autonomous Feature Factory (MAGNET Pattern)

- **Effort:** 2 weeks | **Impact:** 10x feature coverage
- **Audit section:** [`10_ADVANCED_RESEARCH.md` §AR-10](../10_ADVANCED_RESEARCH.md)
- **Concept:** Three phases:
  - **Enumeration** (weekly overnight): For each base feature → lags, rolling stats, cross-interactions, regime-conditional variants, residualized features. ~500+ candidates.
  - **Screening** (weekly overnight): Compute IC for all candidates. Filter IC > 0.01 AND stability > 0.5. Drop >0.95 correlation. Output: ~50-100 curated features.
  - **Monitoring** (daily): PSI drift per feature. IC decay tracking. Auto-replace decaying features.
- **Acceptance criteria:**
  - [ ] Feature enumeration produces 500+ candidates weekly
  - [ ] IC screening filters to 50-100 curated features
  - [ ] Daily monitoring auto-replaces decaying features

### AR-5: Long-Horizon Parallel Research (Kosmos Pattern)

- **Effort:** 2 weeks | **Impact:** 224x weekend research throughput
- **Audit section:** [`10_ADVANCED_RESEARCH.md` §AR-5](../10_ADVANCED_RESEARCH.md)
- **Concept:** 4 parallel research streams (Friday 20:00 → Monday 04:00, 56 hours):
  - Factor Mining: academic papers → testable factors → IC screening
  - Regime Research: historical regime labeling → regime-conditional allocation
  - Cross-Asset Signals: bond-equity-FX-commodity lead-lag relationships
  - Portfolio Construction: alternative optimizers (risk parity, Black-Litterman, HRP)
- **Acceptance criteria:**
  - [ ] 4 parallel weekend research streams operational
  - [ ] ~2,688 experiments per weekend (vs. ~12 today)
  - [ ] Monday morning: curated results ready for validation

---

## Sub-Phase 10C: Intelligence Layer (Week 15-18, ~5 weeks)

### AR-3: Alpha Knowledge Graph (AI-Supervisor Pattern)

- **Effort:** 4 weeks | **Impact:** Institutional memory, factor crowding prevention
- **Audit section:** [`10_ADVANCED_RESEARCH.md` §AR-3](../10_ADVANCED_RESEARCH.md)
- **Concept:** Structured knowledge graph: Factors, Strategies, Hypotheses, Results, Evidence, Regimes as nodes. Edges: uses, tested_by, correlates_with, contradicted_by, favors.
- **Killer feature:** `fund_manager` queries: "Does this strategy share >2 factors with existing position?" Factor-level correlation (smarter than price correlation).
- **Implementation:** PostgreSQL JSON columns, not separate graph DB. `alpha_knowledge_graph` table with node/edge schema.
- **Acceptance criteria:**
  - [ ] Knowledge graph populated with strategies, factors, results
  - [ ] Factor-level correlation available for pre-trade checks
  - [ ] Redundant research detection (already tested this hypothesis)

### AR-6: Consensus-Based Signal Validation

- **Effort:** 1 week | **Impact:** Fewer catastrophic trades
- **Audit section:** [`10_ADVANCED_RESEARCH.md` §AR-6](../10_ADVANCED_RESEARCH.md)
- **Concept:** For high-stakes decisions (>$5K): 3 independent agents (bull advocate, bear advocate, neutral arbiter) evaluate in parallel. Deterministic merge: 3/3 agree → full size; 2/3 → 50%; <2/3 → reject. Low-stakes (<$5K): single agent (current behavior).
- **Acceptance criteria:**
  - [ ] High-stakes trades require consensus from 3 agents
  - [ ] Consensus level determines position sizing
  - [ ] Low-stakes trades bypass (no latency penalty)

---

## Sub-Phase 10D: Self-Improvement (Week 18-24, ~6 weeks)

### AR-2: Metacognitive Self-Modification (DGM-Hyperagents)

- **Effort:** 3 weeks | **Impact:** Compounding quality improvement
- **Audit section:** [`10_ADVANCED_RESEARCH.md` §AR-2](../10_ADVANCED_RESEARCH.md)
- **Concept:** 4 meta agents that improve task agents:
  - `meta_prompt_optimizer`: Analyze outcomes per agent → generate prompt variants → A/B test (weekly)
  - `meta_threshold_tuner`: Adjust confidence thresholds by 0.05 increments based on outcome data (monthly)
  - `meta_tool_selector`: Track tool usage → inject unused-but-needed tools → remove always-failing (weekly)
  - `meta_architecture_critic`: Compare Sharpe to benchmark → identify bottleneck → propose improvement (quarterly)
- **Recursive:** `meta_prompt_optimizer` has its own prompt. If suggestions don't improve performance, `meta_architecture_critic` modifies the optimizer's prompt.
- **Acceptance criteria:**
  - [ ] All 4 meta agents operational
  - [ ] Measurable improvement in task agent performance over 30-day windows
  - [ ] Recursive self-modification demonstrated

### AR-4: Hierarchical Governance (OrgAgent Pattern)

- **Effort:** 3 weeks | **Impact:** 74% token reduction
- **Audit section:** [`10_ADVANCED_RESEARCH.md` §AR-4](../10_ADVANCED_RESEARCH.md), [`05_GRAPH_RESTRUCTURING.md`](../05_GRAPH_RESTRUCTURING.md) (Graph 4: Governance)
- **Concept:** Separate governance (CIO Agent, Sonnet, once/day), execution (strategy agents, Haiku, every 5 min), and compliance (Risk Officer, deterministic, per-trade). CIO at $0.15/day + 4 strategy agents at Haiku ~$3.20/day + deterministic compliance = ~$10/day vs. current $150-450/day.
- **Note:** This overlaps significantly with the 5-graph architecture in [`05_GRAPH_RESTRUCTURING.md`](../05_GRAPH_RESTRUCTURING.md) (Graph 4: Governance). Implementation should be coordinated.
- **Acceptance criteria:**
  - [ ] CIO Agent produces daily mandate constraining trading decisions
  - [ ] Strategy agents operate within mandate using Haiku
  - [ ] Compliance layer is immutable (same protection as risk_gate.py)
  - [ ] Token cost reduced 70%+ vs. current architecture

---

## AutoResearchClaw Upgrades

The existing AutoResearchClaw (`scripts/autoresclaw_runner.py`, 683 lines) is the backbone for AR-1, AR-7, and AR-8. Four upgrades needed:

| Gap | Current | Required | Fix |
|-----|---------|----------|-----|
| **Weekly schedule** | Sunday 20:00 only | Nightly 20:00 | Change scheduler |
| **Reactive only** | Bug_fix and drift only | Proactive gap detection + hypothesis | Add gap detection feeds |
| **tmux restarts** | `tmux send-keys` | Docker Compose restart | Use `docker compose restart` |
| **No functional validation** | py_compile + import only | Tool invocation test after fix | Add tool test after patch |

---

## Dependencies

- **Hard prerequisite:** Phases 1-7 substantially complete (validated signals, safe execution, feedback loops)
- **10A depends on Phase 7** (failure mode taxonomy for AR-7)
- **10B depends on 10A** (budget discipline and tool registry cleanup)
- **10C depends on 10B** (research output to populate knowledge graph)
- **10D depends on 10C** (performance data for meta-optimization)

---

## Cost-Benefit Summary

| Technique | Effort | Annual Token Impact | Alpha Impact |
|-----------|--------|---------------------|--------------|
| Autoresearch (AR-1) | 1 week | -$2K (more research) | 16x hypothesis throughput |
| Meta layer (AR-2) | 3 weeks | +$5K (meta agents) | Compounding quality improvement |
| Knowledge graph (AR-3) | 4 weeks | $0 (deterministic) | Prevent redundancy + factor crowding |
| OrgAgent hierarchy (AR-4) | 3 weeks | **-$50K** (74% token reduction) | Same or better quality |
| Consensus (AR-6) | 1 week | +$3K (3x LLM on big trades) | Fewer catastrophic trades |
| Error-driven (AR-7) | 1 week | $0 | Systematic loss reduction |
| Feature factory (AR-10) | 2 weeks | $0 (CPU compute) | 10x feature coverage |
| Budgets (AR-9) | 1 week | **-$15K** (bounded spend) | Predictable costs |
| Dynamic tools (AR-8) | 1 week | $0 | Self-improving tool layer |
| Parallel research (AR-5) | 2 weeks | +$5K (weekend compute) | 224x weekend throughput |
| **NET** | **~20 weeks** | **~$54K/year savings** | **Compounding alpha + self-improvement** |

---

## The Endgame

```
TODAY (v1.0):  Human writes strategy → System backtests → Trades → Human reviews
PHASE 2 (v2.0): System generates 96 hypotheses/night → Filters → Trades → Logs losses
PHASE 3 (v3.0): Knowledge graph prevents redundancy → Consensus validates → Loss analyzer feeds research
PHASE 4 (v4.0): Meta layer improves hypothesis quality → Tunes thresholds from outcomes →
                 Optimizes its own optimization → Compounds intelligence with capital
```

---

## Validation Plan

Phase 10 validation is ongoing and empirical:

1. **10A:** After 1 week of AR-8: count active vs. stub tools (target: 50+ active). After AR-7: verify loss → research → task pipeline.
2. **10B:** After 1 month of AR-1: count overnight experiments (target: 2000+/month). Measure winner rate (target: 5-10% pass morning validation).
3. **10C:** After AR-3: query knowledge graph for factor overlap. Verify it prevents a known-crowded factor allocation.
4. **10D:** After AR-2 1-month run: compare task agent win rates before/after meta-optimization. Expect 5-10% improvement.
