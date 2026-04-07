# QuantStack CTO Onboarding — Executive Briefing

**Prepared for:** Incoming CTO
**Date:** 2026-04-06
**Compiled from:** Three independent audits — CTO Architecture Audit, Principal Quant Scientist Deep Audit, and Deep Operational Audit
**Classification:** Internal — Confidential

---

## What Is QuantStack?

QuantStack is an autonomous trading company — no humans in the loop. It researches strategies, trains models, executes trades, and learns from outcomes. Three LangGraph StateGraphs run as Docker services:

- **Research Graph** — strategy discovery, ML training, hypothesis validation (7 agents, 10-min cycles)
- **Trading Graph** — position monitoring, entry scanning, execution (10 agents, 5-min cycles, market hours)
- **Supervisor Graph** — health monitoring, self-healing, strategy lifecycle (4 agents, always-on)

21 specialized agents, 16 concurrent signal collectors, PostgreSQL + pgvector backing store, Langfuse observability, multi-provider LLM routing (Anthropic, Bedrock, Groq, Ollama).

---

## Bottom Line Up Front

**The engineering is impressive. The system is not ready for real capital.**

Three independent audits converged on the same conclusion: QuantStack has production-grade architecture wrapped around unvalidated signals, phantom execution, and zero closed feedback loops. The architecture is a B+. The quant substance is a D+. Combined grade: **C-**.

The system can generate alpha in paper trading. Whether that alpha is real or an artifact of optimistic backtesting, look-ahead bias, survivorship bias, and unrealistic execution assumptions — we genuinely don't know. The statistical validation infrastructure to answer that question does not exist yet.

---

## Unified Scorecard

| Subsystem | CTO Audit | Quant Scientist | Deep Ops | Combined | Key Issue |
|-----------|-----------|-----------------|----------|----------|-----------|
| Execution & Risk | B- | D | CRITICAL | **D+** | No real algo execution, stops optional, no Greeks in gate |
| Signal Quality | (not audited) | D- | — | **D-** | No IC computation, no confidence intervals, no decay model |
| Backtesting | (not audited) | D- | — | **D-** | Underapplied costs, survivorship bias, no Monte Carlo |
| ML Pipeline | (not audited) | D | — | **D** | No hyperparameter optimization, weak drift detection |
| Graph Architecture | B | C | — | **C+** | Race conditions, errors don't block execution |
| Data & Signals | B | C- | — | **C** | AV single point of failure, no staleness rejection |
| LLM & Prompts | C+ | C+ | MEDIUM | **C+** | Prompt injection, silent parse failures, 0 caching |
| Ops & Infrastructure | C+ | D+ | HIGH | **D+** | Root containers, exposed DB, no backups, no CI/CD |
| Agent Architecture | — | — | — | **C** | Missing roles, no tool access control, state unvalidated |
| Feedback & Learning | — | F | CRITICAL | **F** | 5 learning modules built, zero wired, losses don't teach |
| Cost Optimization | C | — | — | **C** | 10x overspend on prompt tokens, no caching enabled |
| SEC Compliance | — | — | CRITICAL | **F** | Zero: no wash sale, PDT, margin, tax lot tracking |
| **OVERALL** | **B-** | **D+** | **D+** | **C-** | |

---

## Finding Count (Deduplicated Across All Three Audits)

| Severity | Count | Examples |
|----------|-------|---------|
| **CRITICAL** | 38 | Stop-loss optional, no signal IC, prompt injection, no backups, root containers |
| **HIGH** | 60 | No pre-trade correlation, no intraday circuit breaker, stale signals, no fallback providers |
| **MEDIUM** | 66 | No few-shot examples, feature multicollinearity, no SBOM scanning |
| **TOTAL** | **164** | 5 retracted after code verification (intellectual honesty) |

---

## What's Working Well (Don't Break These)

| Component | Why It's Good |
|-----------|---------------|
| **Signal Engine** | 16 concurrent collectors, 2-6s wall-clock, fault-tolerant, regime-adaptive weights |
| **Risk Gate** | Multi-layer enforcement: daily loss, position caps, gross exposure, options DTE/premium |
| **Kill Switch** | Two-layer design (DB + sentinel file) survives process crashes |
| **Agent Specialization** | 21 agents with clear, non-overlapping roles across 3 graphs |
| **Execution Monitor** | Deterministic exit rules evaluated on every price tick, priority-ordered |
| **Strategy Lifecycle** | Draft → backtested → forward_testing → live → retired with evidence gates |
| **Self-Healing** | AutoResearchClaw patches tool failures, supervisor diagnoses and recovers |
| **Observability** | Langfuse traces every node, LLM call, and tool invocation |
| **SPAN Margin** | Full 16-scenario stress testing exists in `core/risk/span_margin.py` (537 lines) |
| **Options Greeks** | Portfolio delta/gamma/theta/vega tracking in `core/risk/options_risk.py` (444 lines) |
| **Reconciliation** | `guardrails/agent_hardening.py` detects phantom/unknown positions + quantity mismatches |
| **Paper Broker** | Half-spread slippage + sqrt volume impact + partial fills — realistic enough for validation |

---

## The 10 Most Dangerous Findings

These are the findings that, if left unfixed, would cause catastrophic loss of capital or legal liability:

| Rank | Finding | Risk | Audit Source |
|------|---------|------|-------------|
| 1 | **Stop-losses are optional** — LLM can place trades with `stop_price=None`, no downside protection | Position goes to zero | CTO C1 |
| 2 | **No signal IC computation** — zero validation that signals predict returns. Trading on unvalidated signals. | Entire alpha thesis may be noise | QS-S1 |
| 3 | **Prompt injection** — untrusted data injected into LLM prompts via f-strings, no sanitization | Adversarial manipulation of trades | CTO LC1 |
| 4 | **No database backups** — all state in single PostgreSQL, no pg_dump, no WAL archiving | Total data loss on disk failure | CTO OC1 |
| 5 | **Zero SEC compliance** — no wash sale, PDT, margin tracking, or tax lot accounting | Legal liability, account restrictions | DO-8 |
| 6 | **5 learning modules built, zero wired** — losses recorded but never change system behavior | System repeats identical mistakes | DO-1 |
| 7 | **Bracket orders silently degrade** — if broker hiccups, falls back to plain order with zero protection | Unprotected positions in volatile markets | CTO C2 |
| 8 | **PostgreSQL exposed with default password** — port 5434 open, password "quantstack" if .env not set | Full system compromise | QS-I2 |
| 9 | **Containers run as root** — any container compromise gives root privileges | Privilege escalation | QS-I1 |
| 10 | **92 of 122 tools are stubs** — agents offered tools that return errors, wasting LLM round-trips | Silent failures across all agents | CTO TC1 |

---

## The Three Questions That Matter

### 1. "Can this system make money?"

**We don't know.** No signal has ever been validated against forward returns (QS-S1). Backtests use survivorship-biased universes (QS-B2), underapplied transaction costs (QS-B1), and no look-ahead bias protection (QS-S4). Walk-forward validation exists but isn't enforced (QS-B4). The system could be profitable or it could be trading on noise — the infrastructure to distinguish doesn't exist.

### 2. "Can this system trade safely?"

**No.** Stop-losses are optional (C1). Bracket orders silently degrade to plain orders (C2). No intraday circuit breaker on unrealized P&L (QS-E5). Options positions have no Greeks-based risk limits in the gate (QS-E3). Errors from upstream nodes don't block trade execution (QS-A4). The execution layer is a prototype that simulates trading, not a system that can trade.

### 3. "Does this system learn from its mistakes?"

**No.** Five fully-implemented learning modules (`OutcomeTracker`, `SkillTracker`, `ICAttribution`, `ExpectancyEngine`, `StrategyBreaker`) exist with zero consumers. When a trade loses 2%, the loss is recorded in 3 different tables. No downstream system reads those tables. Position sizing the next morning is identical — as if the loss never happened. The system has memory but no learning.

---

## Audit Corrections (Intellectual Honesty)

The Quant Scientist audit initially reported 68 findings. Upon code-level verification, 5 were wrong and retracted:

| Retracted Finding | What Was Claimed | What Actually Exists |
|-------------------|-----------------|---------------------|
| QS-E2: "Zero margin tracking" | No margin awareness | `core/risk/span_margin.py` (537 lines): full SPAN with 16-scenario stress test |
| QS-A1 partial: "No reconciliation" | No broker-vs-system check | `guardrails/agent_hardening.py:463-550`: phantom/unknown position detection |
| QS-I7: "No job overlap detection" | Scheduler starts duplicates | `strategy_lifecycle.py:422-441`: heartbeat guard skips in-progress runs |
| Loop-1: "Zero feedback loops" | No outcome → research path | `hooks/trade_hooks.py:118-144`: loss > 1% → research_queue (incomplete but exists) |
| QS-I5: "No audit log" | No immutable trail | `audit/decision_log.py`: append-only with SHA256 hashes (DB enforcement missing) |

These corrections demonstrate that the codebase is better than a surface scan suggests. The architecture team built real capabilities — the gaps are in wiring and enforcement, not in fundamental design.

---

## The Architectural Blocker: Trading Graph Cannot Complete a Cycle

Beyond the individual findings, a structural analysis revealed that **all 3 graphs are in chronic timeout**. The trading graph's critical path is 1,910s (31.8 minutes) — but the watchdog kills it at 600s. It can never complete. See Section 05 for the full analysis and the 5-graph architecture solution that reduces trading cycle time from 1,910s to ~210s while cutting LLM costs by 80-93%.

---

## How This Document Is Organized

This audit is broken into **13 sections**, ordered by implementation priority:

| Section | File | What It Covers | When to Execute |
|---------|------|----------------|-----------------|
| 00 | This file | Executive briefing, scorecard, top findings | Read first |
| 01 | `01_SAFETY_HARDENING.md` | Existential risks — stops, backups, injection, security | **Week 1-2** |
| 02 | `02_STATISTICAL_VALIDITY.md` | Signal IC, backtesting integrity, ML pipeline rigor | **Week 2-4** |
| 03 | `03_EXECUTION_LAYER.md` | Real algo execution, TCA, Greeks, liquidity, SEC compliance | **Week 2-6** |
| 04 | `04_OPERATIONAL_RESILIENCE.md` | Infrastructure, containerization, CI/CD, monitoring | **Week 3-5** |
| **05** | **`05_GRAPH_RESTRUCTURING.md`** | **3→5 graph split, timeout crisis, 80% cost reduction** | **Week 3-6** |
| 06 | `06_AGENT_ARCHITECTURE.md` | Race conditions, state validation, missing roles, access control | **Week 4-6** |
| 07 | `07_FEEDBACK_LOOPS.md` | The 5 broken loops, ghost modules, wiring plan | **Week 5-7** |
| 08 | `08_COST_OPTIMIZATION.md` | Prompt caching, tool ordering, compaction, memory economics | **Week 3-5** |
| 09 | `09_DATA_SIGNALS.md` | Signal engine, data pipeline, cache freshness, providers | **Week 4-6** |
| 10 | `10_ADVANCED_RESEARCH.md` | AR-1 through AR-10: autoresearch, meta layer, knowledge graph | **Week 8+** |
| 11 | `11_IMPLEMENTATION_ROADMAP.md` | Unified phased plan with all findings ordered | Reference |
| 12 | `12_APPENDIX_ALL_FINDINGS.md` | Complete registry: all MEDIUM findings, tool layer, retracted | Reference |

**Read sections 01-03 and 05 before your first week is over.** They contain the findings that could cause capital loss or legal liability, plus the architectural blocker that prevents the trading graph from completing.

---

## Key Decision Already Made: RAG vs. File-Based Memory

**Verdict from CTO Audit: Stay with file-based memory.** RAG would cost ~8x more per session by destroying prompt cache hits. The `.claude/memory/*.md` approach is architecturally correct at current scale (<50K tokens). OpenClaw (349K stars) validated the same conclusion. This decision is sound — do not revisit unless memory corpus exceeds 50K tokens.

---

## The Path from C- to A

| Phase | Focus | Timeline | Gate |
|-------|-------|----------|------|
| **Phase 1** | Safety hardening — stop the bleeding | Week 1-2 | Every position has a broker-held stop order |
| **Phase 2** | Statistical validation — know if signals work | Week 2-4 | Daily IC computed for all collectors, confidence intervals on conviction |
| **Phase 3** | Operational safety — survive 24/7 | Week 4-6 | Automated backups, CI/CD, containerized scheduler, intraday circuit breakers |
| **Phase 4** | Learning loops — compound intelligence | Week 6-10 | Losses drive research, IC decay adjusts weights, live Sharpe triggers demotion |
| **Phase 5** | Advanced research — become a company | Week 10+ | 96 experiments/night, meta-optimization, knowledge graph, hierarchical governance |

**Fix the statistics. Fix the execution. Close the loops. Then scale.**
