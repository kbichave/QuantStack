---
name: lit_review
description: Research-to-product gap analysis — find techniques that improve alpha discovery, learning loops, or risk management for QuantStack's autonomous trading system.
user_invocable: true
---

# /lit-review — Research-to-Product Gap Analysis

## Purpose

Find research that directly improves QuantStack's ability to:
1. **Discover profitable strategies faster** (alpha discovery)
2. **Adapt before strategies degrade** (learning & drift)
3. **Execute with less slippage** (execution quality)
4. **Size positions dynamically** (risk-aware RL)

This is NOT an academic survey. Every finding must map to a specific
QuantStack module, with a concrete implementation path and expected impact.

Run when: quarterly audit, after a strategy fails unexpectedly, when
exploring a new capability, or when the user asks a targeted question.

---

## What Matters to This Repo

QuantStack is a **swing-trade system** (1–15 day holds) on **US equities + ATM options**,
running **deterministic signal generation** (SignalEngine, no LLM in hot path),
with **LightGBM/XGBoost** models, **3 SAC RL agents in shadow mode**, and a
**Bayesian regime_affinity learning loop**. The universe is small (XOM, MSFT, IBM)
and expanding. Strategies are rule-based with RSI mean-reversion as the core edge.

### Current Weak Points (from workshop_lessons + strategy_registry)
- **Alpha discovery is brute-force:** 4 templates × ~200 param combos = ~800 candidates.
  HypothesisAgent adds LLM-generated ideas but without evolutionary pressure.
- **Drift detection is reactive:** IC must decay before SkillTracker flags it.
  Multi-day lag between degradation start and response.
- **RL agents are stuck in shadow mode:** SAC needs live data, sample-hungry.
  63+ trading days before promotion gate even evaluates.
- **Risk limits are static:** Same 10% position cap in low-vol and high-vol regimes.
- **Sentiment is basic:** Single news score. No earnings call analysis, no SEC filings.
- **Edge is narrow:** All active strategies are RSI<35 mean-reversion variants.
  No momentum, no breakout, no cross-asset, no macro strategies proven yet.

### Where QuantStack Is Already Strong (don't waste time here)
- **Regime detection:** 3 methods (HMM, TFT, BayesianChangepoint) + rule-based. Near frontier.
- **Safety architecture:** risk_gate + kill_switch + MCPResponseValidator + AgentHardening.
  Research is behind us on guardrails for autonomous trading.
- **Signal latency:** SignalEngine (2-6s, deterministic) beats all LLM-as-trader papers.
- **Backtesting rigor:** Walk-forward, MTF, options, sparse-signal-aware. Solid.

---

## Research Domains (Priority Order)

### 1. Alpha Discovery — CRITICAL GAP
**Why it's #1:** All 4 active strategies share the same RSI<35 edge. One regime
shift that breaks mean-reversion wipes the portfolio. We need diverse alpha sources.

**QuantStack today:** `packages/quantstack/alpha_discovery/engine.py` — grid search
over 4 templates + LLM HypothesisAgent. `src/quantstack/core/features/` — 200+ indicators.

**Search for:**
- Grammar-guided GP / symbolic regression for alpha factor discovery
  (AlphaCFG, HARLA, Warm-Start GP)
- RL-based alpha generation (PPO for formulaic alpha sets)
- Causal discovery for factor models (DYNOTEARS, CD-NOTS, CausalFormer)
  to filter spurious correlations from our 200+ features
- Transfer learning across instruments (can XOM alpha transfer to CVX?)
- LLM-in-the-loop evolutionary optimization (formalizing our HypothesisAgent)

**Key question:** Does GP/symbolic regression discover alphas that grid search
misses? If yes, how much IC/IR improvement and at what compute cost?

### 2. Continuous Learning & Concept Drift — CRITICAL GAP
**Why it's #2:** Our learning loop has a multi-day lag. Research shows proactive
drift detection prevents 30-50% of degradation periods.

**QuantStack today:** `packages/quantstack/learning/outcome_tracker.py` — Bayesian
momentum (step=0.05, ~20 trades to move significantly).
`learning/skill_tracker.py` — IC decay detection (reactive).

**Search for:**
- Proactive drift detection (Proceed, DynaME) — detect shift before performance drops
- Recurring vs emergent drift classification — maps to our regime cycling
- Meta-learning for few-shot strategy adaptation (IJCAI-style MAML for trading)
- Adaptive ensemble methods that reweight models without full retraining
- Catastrophic forgetting prevention for continuous RL updates

**Key question:** Can we detect distribution shift in signal features *before*
it shows up as IC degradation? What's the false positive rate?

### 3. RL Architecture Upgrades — SIGNIFICANT GAP
**Why it's #3:** Our 3 SAC agents need 63+ trading days in shadow mode.
Decision Transformers train offline on historical data with no exploration risk.

**QuantStack today:** `src/quantstack/rl/` — 3 independent SAC agents
(execution, sizing, meta). Shadow mode with promotion gating.

**Search for:**
- Decision Transformer / offline RL for trading (pretrained LLM + LoRA initialization)
- Reward shaping to reduce sparsity (trading rewards are sparse and delayed)
- Almgren-Chriss + RL hybrid for execution (10% shortfall improvement reported)
- Risk-sensitive RL with CVaR constraints (dynamic risk budgeting, not static limits)
- Cooperative MARL (our 3 agents don't communicate — should they?)

**Key question:** Does a Decision Transformer trained on our trade history
outperform SAC in shadow mode? What's the minimum history needed?

### 4. Sentiment & Alternative Data — SIGNIFICANT GAP
**Why it's #4:** Our sentiment collector produces a single number from news
headlines. Earnings call analysis alone produced Sharpe 0.13–1.28 in research.

**QuantStack today:** `signal_engine/collectors/sentiment.py` — basic news score.
`collectors/fundamentals.py` — P/E, FCF, ROE.

**Search for:**
- RAG-augmented earnings call / SEC filing analysis (MarketSenseAI 2.0)
- LLM-based earnings call Q&A sentiment (192K transcript study)
- Event-driven sentiment (earnings surprise + management tone → alpha)
- Social media / options flow sentiment integration

**Key question:** Does RAG over earnings calls produce actionable alpha
for our swing-trade universe? What data sources are freely available?

### 5. Foundation Models vs Feature Engineering — INVESTIGATE
**Why it's #5:** Could replace our entire 200+ feature pipeline with zero-shot
prediction. BUT research shows TSFMs underperform domain-specific models on
financial data. Needs empirical validation, not blind adoption.

**QuantStack today:** LightGBM/XGBoost on hand-crafted features. `HierarchicalEnsemble`
for multi-timeframe fusion.

**Search for:**
- Chronos-2, TimesFM, Lag-Llama benchmarks on financial data specifically
- Financial foundation models (Kronos, domain-specific pretraining)
- Papers that compare TSFMs vs gradient-boosted trees on same financial datasets
- Zero-shot vs fine-tuned performance on equity OHLCV

**Key question:** On our exact symbols and labels, does Chronos-2 zero-shot
beat our LightGBM? This is an empirical question, not a literature question.

### 6. Options & Derivatives ML — DEFER UNLESS SCALING
**Why it's lower:** Our BS pricing is adequate for ATM equity options. Neural
pricing matters for exotic options, vol surface arb, or high-frequency flow.
None of those are current QuantStack use cases.

**QuantStack today:** `quantstack/core/options/` — BS pricing, Greeks, IV calc.
`backtesting/options_engine.py` — synthetic IV backtesting.

**Search for (only if options volume is scaling):**
- Deep hedging with IV surface (6x MSE reduction vs delta hedging)
- PINNs for option pricing in illiquid markets
- Vol surface prediction with CNNs
- Options flow signal extraction as alpha

---

## Workflow

### Step 0: Read Context
- Read `.claude/memory/lit_review_findings.md` — prior review findings + roadmap
- Read `.claude/memory/workshop_lessons.md` — what's been tried and failed
- Read `.claude/memory/strategy_registry.md` — current strategy landscape
- Read `.claude/memory/session_handoffs.md` — recent priorities and build status

### Step 1: Scope
State what you're reviewing:
- **Deep-dive:** Single domain (e.g., "Domain 1: Alpha Discovery — GP methods")
- **Multi-domain:** 2-3 related areas (e.g., "Domains 1+2: Discovery + Drift")
- **Full survey:** All domains at headline level (quarterly)
- **Targeted:** Specific question (e.g., "Should we replace SAC with Decision Transformer?")

Check `lit_review_findings.md` — if a domain was reviewed < 3 months ago and
no new strategies have failed in that area, skip it and focus elsewhere.

### Step 2: Search
For each domain in scope, search for recent papers (2023+):

**Query patterns:**
```
"{technique}" trading {sub-topic} arxiv 2024 2025 2026
"{technique}" stock market empirical results Sharpe
"{technique}" vs {our_approach} financial benchmark
```

**Per paper, extract:**
- Title, venue, year
- Core method (1-2 sentences)
- Results (Sharpe, IC, drawdown — whatever they report)
- Does it include transaction costs? (flag if not)
- Is there open-source code?
- Relevance: DIRECT / ADJACENT / THEORETICAL

**Discard papers that:**
- Report only in-sample results
- Assume zero transaction costs on strategies with >100 trades/year
- Test on <5 instruments without cross-validation
- Require tick data or LOB data (we don't have it)
- Require GPU clusters for inference (we have M3 Max, not a cluster)

### Step 3: Gap Analysis
For each domain, produce:

```markdown
### {Domain Name}

**Current:** {QuantStack module + what it does}
**Frontier:** {Best paper result}
**Gap:** CRITICAL / SIGNIFICANT / MINOR / NONE
**Delta:** {What's missing, quantified}

**Specific gaps:**
1. {Gap} — {Paper} shows {result}. We have {current}. Effort: {L/M/H}. Impact: {quantified}.

**Verdict:** BUILD / INVESTIGATE / SKIP / DEFER
```

Severity definitions:
- **CRITICAL:** Research shows 2x+ improvement AND fits our trading frequency/data/compute
- **SIGNIFICANT:** Clear improvement, moderate effort, compatible with our stack
- **MINOR:** Marginal gain or incompatible constraints (tick data, GPU cluster, HFT)
- **NONE:** We're at or ahead of the frontier

### Step 4: Roadmap
Rank all gaps into a priority table:

| Priority | Gap | Effort | Impact | Prerequisite |
|----------|-----|--------|--------|-------------|
| P0 | ... | ... | ... | ... |

Weighting:
1. **Alpha diversification** (35%) — does this reduce our RSI-only concentration?
2. **Degradation prevention** (25%) — does this catch problems before they cost money?
3. **Implementation fit** (25%) — works with swing trades, OHLCV data, M3 Max, LightGBM stack?
4. **Research maturity** (15%) — proven with code, or speculative?

### Step 5: Update Memory + Write Report
- Update `.claude/memory/lit_review_findings.md` with new findings
- Update `.claude/memory/session_handoffs.md` with handoff to /workshop
- Write report to `reports/literature_review/lit_review_{date}.md`

Report structure:
```markdown
# Literature Review: {Focus} — {Date}

## Executive Summary
{3-5 bullets: biggest gaps, highest-impact items, surprises}

## Findings by Domain
{Step 3 output}

## Roadmap
{Step 4 table}

## Papers
{Numbered list with links}

## Next Steps
{What to build in /workshop, what to investigate, what to skip}
```

---

## Quality Filters

### Paper Must-Haves
- Empirical results with out-of-sample testing
- Transaction costs included (or explicitly flagged if not)
- Tested on >5 instruments or >3 years of data
- 2023+ publication (older only if foundational and nothing newer exists)

### QuantStack Compatibility Check
Before recommending BUILD on any technique, verify:
- [ ] Works at swing-trade frequency (1-15 day holds)?
- [ ] Works with daily/hourly OHLCV + fundamentals data?
- [ ] Runs on M3 Max (128GB unified memory, no GPU cluster)?
- [ ] Integrates with regime-gated strategy routing?
- [ ] Compatible with risk_gate position limits and daily loss limits?
- [ ] Doesn't require real-time tick/LOB data?
- [ ] Implementation < 2 weeks for P0, < 4 weeks for P1?

### What NOT to Recommend
- LLM-as-trader architectures (we moved past this — SignalEngine is faster and deterministic)
- Techniques requiring tick data or limit order book data
- GPU-cluster-only training (must fit M3 Max)
- Theory-only papers without empirical validation on equities
- Techniques that only work for HFT (<1 minute holding periods)

---

## Incremental Review Protocol

After the first full survey, subsequent sessions:
1. Read prior `lit_review_findings.md`
2. Search only for papers published since last review date
3. Check which roadmap items were built — did they deliver expected impact?
4. Re-rank remaining items based on what changed in strategy_registry
5. If a strategy failed for a reason research could have predicted, escalate that domain

---

## Notes
- This skill produces RESEARCH, not code. Implementation goes to /workshop.
- P0 findings → immediate handoff note in `session_handoffs.md` for /workshop.
- If a finding contradicts a QuantStack design choice, check CLAUDE.md invariants
  before recommending changes. The current design may exist for safety reasons.
- Report goes in `reports/literature_review/` alongside prior reviews.
