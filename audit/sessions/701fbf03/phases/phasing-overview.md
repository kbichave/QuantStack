# Phase Overview: QuantStack → Autonomous Fund

**Date:** 2026-04-07
**Baseline:** Post-CTO-audit implementation (169 findings addressed)
**Vision:** Autonomous trading company, no humans, Harvard-IB-grade
**Capital:** <$100K personal
**Approach:** Full parallel, dependency-graph ordered

---

## Dependency Graph

```
P00 (Wire Learning) ──────────────────────────────────────────┐
P01 (Signal Rigor) ──→ P05 (Adaptive Synthesis) ─────────────┤
P02 (Execution Reality) ──→ P08 (Options MM) ────────────────┤
P03 (ML Pipeline) ──→ P09 (RL Trading) ──────────────────────┤──→ P15 (Autonomous Fund)
P04 (Backtest Integrity) ─────────────────────────────────────┤
P05 ──→ P10 (Meta-Learning) ──────────────────────────────────┤
P06 (Options Desk) ──→ P08 ──────────────────────────────────┤
P07 (Data Evolution) ──→ P11 (Alt Data) ─────────────────────┤
P08 ──→ P12 (Multi-Asset) ───────────────────────────────────┤
P09 ──────────────────────────────────────────────────────────┤
P10 ──────────────────────────────────────────────────────────┤
P11 ──────────────────────────────────────────────────────────┤
P12 ──────────────────────────────────────────────────────────┤
P13 (Causal Alpha) ───────────────────────────────────────────┤
P14 (Advanced ML) ────────────────────────────────────────────┘
```

## Milestones

### M1: "The System Learns" (P00 + P01 + P04)
**Phases:** P00-Wire-Learning-Modules, P01-Signal-Rigor, P04-Backtest-Integrity
**What ships:** Signals validated against returns. Feedback loops closed. Backtests trustworthy.
**Why first:** Everything downstream depends on knowing whether signals predict returns.

### M2: "The System Trades for Real" (P02 + P03 + P06)
**Phases:** P02-Execution-Reality, P03-ML-Pipeline, P06-Options-Desk
**What ships:** Real execution algos, functioning ML, Greeks-aware options.
**Can run parallel with M1.**

### M3: "The System Improves Itself" (P05 + P07 + P10)
**Phases:** P05-Adaptive-Synthesis, P07-Data-Evolution, P10-Meta-Learning
**What ships:** IC-driven weight adjustment, multi-provider data, prompt optimization.
**Depends on:** M1 (needs IC tracking to drive adaptation).

### M4: "The Sophisticated Fund" (P08 + P09 + P11)
**Phases:** P08-Options-Market-Making, P09-RL-Trading, P11-Alternative-Data
**What ships:** Vol arb, RL portfolio optimization, alt data signals.
**Depends on:** M2 (needs real execution + ML pipeline).

### M5: "The Multi-Asset Company" (P12 + P13 + P14 + P15)
**Phases:** P12-Multi-Asset, P13-Causal-Alpha, P14-Advanced-ML, P15-Autonomous-Fund
**What ships:** Futures/forex/crypto, causal inference, transformers, full autonomy.
**Depends on:** M3 + M4.

## Parallelism Opportunities

| Parallel Group | Phases | Why Parallel |
|---------------|--------|-------------|
| Foundation | P00 + P01 + P02 + P03 + P04 | No cross-dependencies — different subsystems |
| Enhancement | P05 + P06 + P07 | Different subsystems, no shared files |
| Expansion | P08 + P09 + P11 | Independent capabilities |
| Advanced | P12 + P13 + P14 | Independent research areas |

## Phase Summary

| Phase | Name | Depends On | Effort | Priority |
|-------|------|-----------|--------|----------|
| P00 | Wire Learning Modules | None | 2-3 days | CRITICAL |
| P01 | Signal Statistical Rigor | None | 1-2 weeks | CRITICAL |
| P02 | Execution Reality | None | 1-2 weeks | CRITICAL |
| P03 | ML Pipeline Completion | None | 1-2 weeks | CRITICAL |
| P04 | Backtest Integrity | None | 1 week | CRITICAL |
| P05 | Adaptive Signal Synthesis | P01 | 1 week | HIGH |
| P06 | Options Desk Upgrade | None | 1-2 weeks | HIGH |
| P07 | Data Architecture Evolution | None | 1 week | HIGH |
| P08 | Options Market-Making | P02, P06 | 2-3 weeks | MEDIUM |
| P09 | Reinforcement Learning Pipeline | P03 | 2-3 weeks | MEDIUM |
| P10 | Meta-Learning & Self-Improvement | P05 | 2 weeks | MEDIUM |
| P11 | Alternative Data Sources | P07 | 2 weeks | MEDIUM |
| P12 | Multi-Asset Expansion | P02, P07 | 3-4 weeks | LOW |
| P13 | Causal Alpha Discovery | P01, P03 | 2 weeks | LOW |
| P14 | Advanced ML (Transformers, GNNs) | P03 | 2-3 weeks | LOW |
| P15 | Autonomous Fund Integration | ALL | 2 weeks | FINAL |
