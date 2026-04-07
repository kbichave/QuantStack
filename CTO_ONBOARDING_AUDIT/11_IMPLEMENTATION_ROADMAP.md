# 10 — Implementation Roadmap: All 164 Findings Unified and Ordered

**This is the master reference.** Every finding from all three audits, deduplicated, ordered by execution priority, with effort estimates and dependencies.

---

## Phase 0: Quick Wins (Day 1-2)

Items that take less than 1 day each and have outsized impact.

| # | Finding | Source | Effort | Impact |
|---|---------|--------|--------|--------|
| 0.1 | Deterministic tool ordering | CTO MC1 | 30 min | 30-50% prompt cost savings |
| 0.2 | Enable prompt caching | CTO MC0c | 1 hour | 10x reduction in system prompt cost |
| 0.3 | Fix `search_knowledge_base` to use RAG | CTO MC0 | 1 hour | Agents get relevant context, not random recent |
| 0.4 | Add HNSW index on embeddings | CTO MC0b | 30 min | RAG queries stay fast at scale |
| 0.5 | Sentiment fallback: `{}` not 0.5 | CTO DH3 | 30 min | No fake neutral signals |
| 0.6 | Bind PostgreSQL to localhost | QS-I2 | 15 min | Close security hole |
| 0.7 | Remove default DB password | QS-I2 | 15 min | Close security hole |

**Total: 1 day. Annual savings from 0.1+0.2 alone: $32,000-$45,000.**

---

## Phase 1: Safety Hardening (Week 1-2)

**Gate: Every position has a broker-held stop order. DB backed up. Prompts sanitized.**

| # | Finding | Source | Section | Effort | Depends On |
|---|---------|--------|---------|--------|-----------|
| 1.1 | Mandatory stop-loss enforcement | CTO C1, C2, C3, DO-2 | 01 | 2-3 days | — |
| 1.2 | Automated DB backups (pg_dump → S3) | CTO OC1 | 01 | 1 day | — |
| 1.3 | Prompt injection defense | CTO LC1 | 01 | 2-3 days | — |
| 1.4 | Output schema validation with retry | CTO LC2 | 01 | 2 days | — |
| 1.5 | Run containers as non-root | QS-I1 | 01 | 0.5 day | — |
| 1.6 | Durable checkpoints (PostgresSaver) | CTO GC1 | 01 | 1 day | — |
| 1.7 | Trading graph polls EventBus | CTO AC1, AC2 | 01 | 1 day | — |
| 1.8 | Kill switch publishes to EventBus | CTO AC2 | 01 | 0.5 day | — |
| 1.9 | Containerize scheduler | CTO OC2 | 01 | 1 day | — |
| 1.10 | DB transaction isolation for positions | QS-I3 | 04 | 1 day | — |

**Total: 12-14 days. Parallelizable to ~8 days with 2 engineers.**

---

## Phase 2: Statistical Validity (Week 2-4)

**Gate: Daily IC computed. Backtests trustworthy. Walk-forward mandatory.**

| # | Finding | Source | Section | Effort | Depends On |
|---|---------|--------|---------|--------|-----------|
| 2.1 | Signal IC computation + tracking | QS-S1 | 02 | 3-4 days | — |
| 2.2 | Signal confidence intervals | QS-S2 | 02 | 2 days | 2.1 |
| 2.3 | Signal decay modeling | QS-S3 | 02 | 2 days | 2.1 |
| 2.4 | Look-ahead bias detection | QS-S4 | 02 | 2-3 days | — |
| 2.5 | Conviction-scaled position sizing | QS-S6 | 02 | 1 day | 2.2 |
| 2.6 | Walk-forward validation gate | QS-B4 | 02 | 2 days | — |
| 2.7 | Survivorship bias adjustment | QS-B2 | 02 | 2 days | — |
| 2.8 | Realistic transaction costs (30 bps) | QS-B1 | 02 | 1 day | — |
| 2.9 | Hyperparameter optimization (optuna) | QS-M1 | 02 | 2-3 days | — |
| 2.10 | Feature multicollinearity audit | QS-B6 | 02 | 2 days | — |
| 2.11 | Monte Carlo validation | QS-B3 | 02 | 2 days | 2.6 |
| 2.12 | Options Greeks in risk gate | QS-E3 | 03 | 3 days | — |
| 2.13 | Intraday circuit breaker | QS-E5 | 03 | 2 days | — |
| 2.14 | Fed event enforcement in risk gate | DO-4 | 03 | 1 day | — |

**Total: 26-32 days. Parallelizable to ~14 days with 2 engineers.**

---

## Phase 3: Operational Resilience (Week 3-5)

**Gate: CI/CD active. Logs aggregated. Kill switch auto-recovers.**

| # | Finding | Source | Section | Effort | Depends On |
|---|---------|--------|---------|--------|-----------|
| 3.1 | Enable CI/CD pipeline | CTO OC3 | 04 | 2 days | — |
| 3.2 | Log aggregation + alerting | CTO OH2 | 04 | 3 days | — |
| 3.3 | Kill switch auto-recovery | CTO OH3 | 04 | 2 days | 1.7, 1.8 |
| 3.4 | Migration versioning | QS-I4 | 04 | 1 day | — |
| 3.5 | Langfuse retention cleanup | CTO OH1 | 04 | 0.5 day | — |
| 3.6 | OOM monitoring | CTO OH5 | 04 | 1 day | — |
| 3.7 | Health check granularity | CTO OH4 | 04 | 2 days | — |
| 3.8 | Shared rate limiter | DO-9 | 04 | 1 day | — |
| 3.9 | Secrets management | CTO (MEDIUM) | 04 | 1 day | — |
| 3.10 | SBOM scanning | QS-I8 | 04 | 0.5 day | 3.1 |
| 3.11 | Env var type validation | CTO (MEDIUM) | 04 | 0.5 day | — |

**Total: 14-15 days. Parallelizable to ~8 days with 2 engineers.**

---

## Phase 4: Agent Architecture & Coordination (Week 4-6)

**Gate: Race conditions fixed. Errors block execution. Tool access controlled.**

| # | Finding | Source | Section | Effort | Depends On |
|---|---------|--------|---------|--------|-----------|
| 4.1 | Race condition fix (parallel branches) | QS-A2 | 05 | 1-2 days | — |
| 4.2 | Errors block execution | QS-A4 | 05 | 1 day | — |
| 4.3 | State schema validation (Pydantic) | QS-A3 | 05 | 2 days | — |
| 4.4 | Node circuit breaker | QS-A5 | 05 | 1 day | — |
| 4.5 | Tool access control per graph | QS-A9, CTO OC-5 | 05 | 1 day | — |
| 4.6 | Event bus cursor atomicity | QS-A7 | 05 | 0.5 day | — |
| 4.7 | Dead letter queue | QS-A8 | 05 | 1 day | — |
| 4.8 | Priority-based message pruning | QS-A6 | 05 | 2 days | — |
| 4.9 | Pre-trade correlation check | CTO H1 | 03 | 1 day | — |
| 4.10 | Portfolio heat budget | CTO H3 | 03 | 1 day | — |
| 4.11 | Sector concentration pre-trade | CTO H4 | 03 | 0.5 day | — |
| 4.12 | Regime flip forced review | CTO H5 | 03 | 1 day | — |

**Total: 13-15 days. Parallelizable to ~7 days with 2 engineers.**

---

## Phase 5: Cost Optimization (Week 3-5, parallel with Phase 3)

**Gate: Per-agent cost tracked. Context compaction active. Tiers assigned.**

| # | Finding | Source | Section | Effort | Depends On |
|---|---------|--------|---------|--------|-----------|
| 5.1 | Context compaction at merge points | CTO MC2, OC-3 | 07 | 3 days | — |
| 5.2 | Per-agent cost tracking | CTO LH4 | 07 | 2 days | — |
| 5.3 | Agent tier reclassification | CTO LH1 | 07 | 1 day | — |
| 5.4 | Per-agent temperature config | CTO LC3 | 07 | 0.5 day | — |
| 5.5 | EWF deduplication | CTO LH5 | 07 | 0.5 day | — |
| 5.6 | Remove hardcoded model strings | CTO LH3 | 07 | 1 day | — |
| 5.7 | LLM provider runtime fallback | CTO LH2, OC-4 | 07 | 2 days | — |
| 5.8 | Memory temporal decay | CTO MC3 | 07 | 1 day | — |

**Total: 11 days. Annual savings: $40,000-$64,000.**

---

## Phase 6: Execution Layer Completion (Week 4-7)

**Gate: Real TWAP/VWAP. TCA feedback. SEC compliance basics.**

| # | Finding | Source | Section | Effort | Depends On |
|---|---------|--------|---------|--------|-----------|
| 6.1 | TCA feedback loop (EWMA) | QS-E6, Loop-2 | 03 | 2 days | — |
| 6.2 | Partial fill tracking | QS-E9 | 03 | 1 day | — |
| 6.3 | Real TWAP/VWAP execution | QS-E1 | 03 | 5 days | — |
| 6.4 | Liquidity model | QS-E4 | 03 | 3 days | — |
| 6.5 | SEC compliance (wash sale, PDT, tax lots) | DO-8 | 03 | 5-7 days | — |
| 6.6 | Best execution audit trail | QS-E7 | 03 | 2 days | — |
| 6.7 | Options monitoring rules | DO-3 | 03 | 2 days | 2.12 |
| 6.8 | Slippage model enhancement | QS-E8 | 03 | 2 days | — |
| 6.9 | Borrowing/funding cost model | QS-E11 | 03 | 1 day | — |

**Total: 23-27 days.**

---

## Phase 7: Feedback Loops & Learning (Week 6-10)

**Gate: Losses drive research. IC decay adjusts weights. System learns.**

| # | Finding | Source | Section | Effort | Depends On |
|---|---------|--------|---------|--------|-----------|
| 7.1 | Wire 6 ghost module readpoints | DO-1 | 06 | 2-3 days | — |
| 7.2 | Failure mode taxonomy | Loop-1 | 06 | 2 days | 7.1 |
| 7.3 | IC degradation → weight adjustment | Loop-3 | 06 | 2 days | 2.1 |
| 7.4 | Live vs. backtest Sharpe demotion | Loop-4 | 06 | 1 day | 2.6 |
| 7.5 | Agent decision quality tracking | Loop-5 | 06 | 3 days | — |
| 7.6 | Loss aggregation in supervisor | DO-1 | 06 | 1 day | 7.2 |
| 7.7 | Signal correlation tracking | QS-S5 | 08 | 2 days | 2.1 |
| 7.8 | Conflicting signal resolution | QS-S9 | 08 | 1 day | — |
| 7.9 | Conviction calibration (multiplicative) | QS-S8 | 08 | 2 days | 2.1 |
| 7.10 | Concept drift: IC + label + interaction | QS-M2 | 02 | 2 days | 2.1 |
| 7.11 | Model versioning + A/B | QS-M3 | 02 | 2 days | — |
| 7.12 | Regime transition detection | QS-S7 | 08 | 3 days | — |

**Total: 23-26 days.**

---

## Phase 8: Data Pipeline Hardening (Week 4-6, parallel with Phase 4-5)

**Gate: Signal cache fresh. Data providers redundant. Intel sources live.**

| # | Finding | Source | Section | Effort | Depends On |
|---|---------|--------|---------|--------|-----------|
| 8.1 | Signal cache auto-invalidation | CTO DC1 | 08 | 1 day | — |
| 8.2 | Staleness rejection in collectors | CTO DC3 | 08 | 2 days | — |
| 8.3 | AV redundancy (FRED, EDGAR) | CTO DC2 | 08 | 5-7 days | — |
| 8.4 | Drift detection pre-cache | CTO DH1 | 08 | 1 day | — |
| 8.5 | Web search configuration | DO-5 | 08 | 0.5 day | — |
| 8.6 | SEC filings population | DO-5 | 08 | 2 days | — |
| 8.7 | OHLCV partitioning | CTO (MEDIUM) | 08 | 2 days | — |
| 8.8 | Options refresh expansion | CTO DH2 | 08 | 1 day | — |

**Total: 14-17 days.**

---

## Phase 9: Missing Roles & Scale (Week 6-8)

| # | Finding | Source | Section | Effort | Depends On |
|---|---------|--------|---------|--------|-----------|
| 9.1 | Corporate actions monitor | QS-A1 | 05 | 2 days | — |
| 9.2 | Factor exposure monitor | QS-A1 | 05 | 1 day | — |
| 9.3 | Performance attribution per-cycle | QS-A1 | 05 | 2 days | — |
| 9.4 | Implement 5 alert lifecycle tools | CTO AC3 | 01 | 3 days | — |
| 9.5 | Real-time Discord for CRITICAL events | CTO AC4 | 01 | 1 day | — |
| 9.6 | Reply-back ACK pattern (EventBus) | CTO AC5 | 01 | 2 days | 1.7 |
| 9.7 | Multi-mode operation (24/7) | CTO 24/7 | 04 | 3-5 days | Phases 1-3 |
| 9.8 | Unify LLM provider systems | DO-7 | 07 | 2 days | 5.6 |
| 9.9 | Research fan-out default on | CTO (MEDIUM) | 04 | 0.5 day | — |

**Total: 17-20 days.**

---

## Phase 10: Advanced Research (Week 10+)

See Section 09 for full details on AR-1 through AR-10.

| Phase | Techniques | Timeline | Effort |
|-------|-----------|----------|--------|
| 10A: Foundation | Registry cleanup (AR-8), Error-driven iteration (AR-7), Experiment budgets (AR-9) | Week 10-11 | 3 weeks |
| 10B: Research Multiplier | Autoresearch loop (AR-1), Feature factory (AR-10), Weekend parallel (AR-5) | Week 11-15 | 5 weeks |
| 10C: Intelligence Layer | Knowledge graph (AR-3), Consensus validation (AR-6) | Week 15-18 | 5 weeks |
| 10D: Self-Improvement | Meta layer (AR-2), Hierarchical governance (AR-4) | Week 18-24 | 6 weeks |

---

## Summary: Total Effort by Phase

| Phase | Calendar | Effort (1 eng) | Effort (2 eng) | Cumulative |
|-------|----------|----------------|----------------|------------|
| Phase 0: Quick Wins | Day 1-2 | 1 day | 1 day | 1 day |
| Phase 1: Safety | Week 1-2 | 13 days | 8 days | 9 days |
| Phase 2: Statistics | Week 2-4 | 28 days | 14 days | 23 days |
| Phase 3: Ops Resilience | Week 3-5 | 15 days | 8 days | 31 days |
| Phase 4: Agent Arch | Week 4-6 | 14 days | 7 days | 38 days |
| Phase 5: Cost Opt | Week 3-5 | 11 days | 6 days | 44 days |
| Phase 6: Execution | Week 4-7 | 25 days | 13 days | 57 days |
| Phase 7: Feedback Loops | Week 6-10 | 24 days | 12 days | 69 days |
| Phase 8: Data Pipeline | Week 4-6 | 15 days | 8 days | 77 days |
| Phase 9: Scale | Week 6-8 | 18 days | 10 days | 87 days |
| Phase 10: Advanced | Week 10+ | ~100 days | 50 days | 137 days |

**With 2 engineers:** Phase 1-9 complete in ~12 weeks. Phase 10 is ongoing.
**With 1 engineer:** Phase 1-9 complete in ~20 weeks. Phase 10 is ongoing.

---

## Critical Path

```
Phase 0 (Day 1) ──→ Phase 1 (Week 1-2) ──→ Phase 2 (Week 2-4)
                          ↓                        ↓
                    Phase 3 (Week 3-5)      Phase 7 (Week 6-10)
                          ↓                        ↓
                    Phase 4 (Week 4-6)      Phase 10 (Week 10+)
                          ↓
                    Phase 5 (parallel with 3)
                          ↓
                    Phase 6 (Week 4-7)
                          ↓
                    Phase 8 (parallel with 4-5)
                          ↓
                    Phase 9 (Week 6-8)
```

**Phase 1 is the hard dependency.** Nothing else is safe to build until safety hardening is complete. Phases 3-5-8 can run in parallel. Phase 7 depends on Phase 2 (need IC tracking to close feedback loops). Phase 10 depends on Phases 1-7.

---

## Annual Financial Impact

| Category | Current Cost/Risk | After Fixes | Net Impact |
|----------|------------------|-------------|------------|
| Prompt token costs | ~$126/day ($46K/yr) | ~$12.60/day ($4.6K/yr) | **-$41K/yr** |
| Agent tier waste | ~$15K/yr | ~$3K/yr | **-$12K/yr** |
| Context bloat | ~$5K/yr | ~$2K/yr | **-$3K/yr** |
| OrgAgent (Phase 10D) | $55-165K/yr | ~$3.6K/yr | **-$50-160K/yr** |
| Total cost savings | | | **$106-216K/yr** |
| Prevented capital loss (stops, circuit breakers) | Unquantified but existential | | |
| Legal risk reduction (SEC compliance) | Unquantified but existential | | |

---

## Appendix Reference

All remaining findings not assigned to specific phase items above — including tool layer findings (TC2, TC3, TH1-TH5) and ~42 unnamed MEDIUM table findings from the original audit — are catalogued with section assignments in [11_APPENDIX_ALL_FINDINGS.md](11_APPENDIX_ALL_FINDINGS.md). These should be addressed as part of their respective phase work.
