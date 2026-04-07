# Implementation Specs — `/deep-plan` Ready

Each spec below is a self-contained implementation blueprint for one phase of the CTO Onboarding Audit. Every spec references the full audit context, lists findings with severity/effort, specifies key files, defines acceptance criteria, identifies dependencies, and includes a validation plan.

**Source audit:** [`CTO_ONBOARDING_AUDIT/`](../README.md) — 164 findings (38 CRITICAL, 60 HIGH, 66 MEDIUM), overall grade C-

---

## Specs by Phase

| Phase | Spec | Timeline | Effort | Key Gate |
|-------|------|----------|--------|----------|
| **0** | [Quick Wins](phase_0_quick_wins.md) | Day 1-2 | 1 day | $32K-$45K/yr savings from tool ordering + caching |
| **1** | [Safety Hardening](phase_1_safety_hardening.md) | Week 1-2 | 8-14 days | Every position has broker-held stop order |
| **2** | [Statistical Validity](phase_2_statistical_validity.md) | Week 2-4 | 14-32 days | Daily IC computed, backtests trustworthy |
| **3** | [Operational Resilience](phase_3_operational_resilience.md) | Week 3-5 | 8-15 days | CI/CD active, logs aggregated, kill switch auto-recovers |
| **4** | [Agent Architecture](phase_4_agent_architecture.md) | Week 4-6 | 7-15 days | Race conditions fixed, errors block execution |
| **5** | [Cost Optimization](phase_5_cost_optimization.md) | Week 3-5 | 11 days | $40K-$64K/yr savings, prompt cache >80% hit rate |
| **6** | [Execution Layer](phase_6_execution_layer.md) | Week 4-7 | 23-27 days | Real TWAP/VWAP, SEC compliance basics |
| **7** | [Feedback Loops](phase_7_feedback_loops.md) | Week 6-10 | 23-26 days | Losses drive research, IC decay adjusts weights |
| **8** | [Data Pipeline](phase_8_data_pipeline.md) | Week 4-6 | 14-17 days | Signal cache fresh, providers redundant |
| **9** | [Missing Roles & Scale](phase_9_missing_roles_scale.md) | Week 6-8 | 17-20 days | Corporate actions monitored, 24/7 mode |
| **10** | [Advanced Research](phase_10_advanced_research.md) | Week 10+ | ~100 days | Self-improving trading company |

---

## Critical Path

```
Phase 0 (Day 1) ──→ Phase 1 (Week 1-2) ──→ Phase 2 (Week 2-4)
                          │                        │
                          ├─ Phase 3 (Week 3-5)    └─ Phase 7 (Week 6-10)
                          │       │                        │
                          │       └─ Phase 5 (parallel)    └─ Phase 10 (Week 10+)
                          │
                          └─ Phase 4 (Week 4-6)
                                  │
                                  ├─ Phase 6 (Week 4-7)
                                  │
                                  └─ Phase 8 (parallel with 4-5)
                                          │
                                          └─ Phase 9 (Week 6-8)
```

**Phase 1 is the hard dependency.** Nothing else is safe until safety hardening is complete. Phases 3/5/8 can run in parallel. Phase 7 depends on Phase 2 (IC tracking). Phase 10 depends on Phases 1-7.

---

## How to Use with `/deep-plan`

Each spec is structured for direct use with `/deep-plan`:

1. **Pick a phase** based on the critical path above
2. **Run `/deep-plan`** referencing the spec file — it contains all context needed
3. **The spec provides:** finding IDs, severity, effort estimates, key files, acceptance criteria, dependencies, risks, and validation plans
4. **The audit sections** (linked from each spec) provide the full technical detail if deeper context is needed

---

## Total Effort Summary

| Engineer Count | Phases 0-9 | Phase 10 | Total |
|---------------|-----------|----------|-------|
| 1 engineer | ~20 weeks | ~20 weeks | ~40 weeks |
| 2 engineers | ~12 weeks | ~10 weeks | ~22 weeks |

**Annual financial impact:** $106K-$216K in cost savings + prevented capital loss + legal risk reduction.
