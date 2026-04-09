# Tier 3-4 Gaps: Differentiators & Future Edge

**Date:** 2026-04-07
**Baseline:** Post-CTO audit (169 findings implemented)

---

## Tier 3: DIFFERENTIATORS — What Separates a Top Fund

These capabilities distinguish elite quantitative funds from standard systematic traders. Each one opens a new source of alpha or a new market.

### G12: Reinforcement Learning Pipeline (→ P09)

**Current:** FinRL is installed and referenced. 11 RL tools are in PLANNED_TOOLS. Zero are functional. No RL environments, no trained policies, no integration with execution.

**What this enables:** RL can learn optimal execution timing, portfolio rebalancing, and strategy selection directly from market interaction — discovering policies that no human would design.

**Build-vs-buy:** Start with FinRL (already installed), evaluate ElegantRL for GPU optimization. Three environments needed:
1. Portfolio optimization (state: positions + signals → action: target weights)
2. Order execution (state: order book + time → action: slice timing)
3. Strategy selection (state: regime + performance → action: strategy allocation)

---

### G13: Options Market-Making (→ P08)

**Current:** Options trading is directional only — buy calls/puts based on signal conviction. No vol arb, no dispersion trading, no market-making, no gamma scalping.

**What this enables:** Market-making strategies profit from bid-ask spread and vol mispricing, generating returns uncorrelated with directional alpha. Dispersion trading exploits index-vs-component vol dislocations.

**Key strategies:**
- Delta-neutral vol selling (short straddles with delta hedging)
- Vol arbitrage (realized vs implied vol divergence)
- Dispersion trading (sell index vol, buy component vol)
- Gamma scalping (long options, trade underlying to capture gamma)
- Pin risk management near expiration

---

### G14: Alternative Data Sources (→ P11)

**Current:** Exclusively traditional data — price, volume, fundamentals, earnings, sentiment (text-based). No unconventional data sources.

**What this enables:** Alpha from data sources most retail/small funds don't access. Congressional trades have shown documented edge (STOCK Act research).

**Priority sources (by cost and alpha potential):**
1. **Quiver Quantitative** — Congressional trades, lobbying data (FREE)
2. **USPTO** — Patent filings as leading indicator of R&D success (FREE)
3. **SimilarWeb** — Web traffic as revenue proxy ($100+/mo)
4. **Job postings** — Hiring patterns signal growth/contraction (scraping or Indeed API)
5. **Satellite/shipping** — Defer until capital > $500K (expensive, marginal at small scale)

---

### G15: Multi-Asset Expansion (→ P12)

**Current:** Equity + equity options only. No futures, forex, crypto, or fixed income.

**What this enables:** True diversification across uncorrelated asset classes. Futures provide leverage-efficient macro exposure. Crypto offers 24/7 trading for overnight compute utilization.

**Priority expansion:**
1. **Crypto spot** — 24/7 markets, fills overnight compute gap (Binance/Coinbase API)
2. **Equity index futures** — E-mini S&P, Nasdaq (CME via IBKR) for macro hedging
3. **Forex** — Major pairs as regime/macro signals (OANDA)
4. **Fixed income ETFs** — TLT, HYG for cross-asset correlation signals

---

### G16: Meta-Learning & Self-Improvement (→ P10)

**Current:** Agent prompts are static. No tracking of which agents make better decisions over time. No automatic prompt optimization.

**What this enables:** The system literally improves its own cognition. OPRO (Optimization by PROmpting) uses LLM-generated prompt variants and selects winners based on measurable outcomes.

**Key components:**
- Agent quality scorecards (accuracy, latency, cost per decision)
- OPRO prompt optimization (generate N variants, A/B test, promote winner)
- Strategy-of-strategies (meta-allocator across strategy families)
- Autonomous research prioritization (highest-ROI research first)

---

### G17: Causal Alpha Discovery (→ P13)

**Current:** All signals are correlation-based. No causal inference, no treatment effect estimation, no counterfactual analysis.

**What this enables:** Causal signals are more regime-stable than correlation-based signals. "Earnings revisions cause price moves" survives regime changes; "momentum correlates with returns" doesn't.

**Key packages:** DoWhy (causal graphs), CausalML (treatment effects), EconML (double ML)

---

### G18: Conformal Prediction (→ P14)

**Current:** ML models produce point predictions. No prediction intervals, no calibrated uncertainty.

**What this enables:** "Model predicts AAPL +2% with 90% CI [+0.5%, +3.5%]" — enables proper position sizing proportional to confidence. A tight CI gets larger size; a wide CI gets reduced size.

**Package:** MAPIE — wraps existing LightGBM/XGBoost directly, scikit-learn compatible.

---

## Tier 4: FUTURE EDGE — Competitive Advantages for Scale

These are cutting-edge capabilities that provide edge at scale. Lower priority but high long-term value.

### G19: Graph Neural Networks for Market Structure (→ P14)

**What this enables:** Model sector/industry relationships, supply chain links, and correlation structure as a graph. GNN provides "sector contagion" signal — if a key supplier drops, predict downstream effect before the market prices it in.

**Packages:** torch_geometric, dgl

---

### G20: Deep Hedging (→ P14)

**What this enables:** Neural network learns optimal hedging strategy for complex option portfolios, handling transaction costs, discrete hedging intervals, and market frictions that Black-Scholes ignores. Replace rule-based delta hedging with a learned policy.

**Research:** Buehler et al. (2019) — proven approach, custom implementation required.

---

### G21: Transformer Forecasting (→ P14)

**What this enables:** Attention-based time series forecasting (PatchTST, iTransformer, TFT). Multi-step ahead predictions with learned temporal context — captures long-range dependencies that traditional ML misses.

**Package:** NeuralForecast (Nixtla) — includes PatchTST, iTransformer, TFT in one package.

---

### G22: Market Microstructure ML (→ P02, P08)

**What this enables:** Order flow toxicity detection (VPIN), Kyle's lambda estimation, informed trader detection. Enables the execution system to detect adverse selection and adjust timing/sizing accordingly.

---

### G23: Domain-Specific Financial NLP (→ P14)

**What this enables:** Beyond generic FinBERT to fine-tuned financial LLMs for SEC filing analysis, earnings call transcripts, and analyst report parsing. Marginal improvement over current sentiment, but compounds with scale.

**Options:** FinGPT, fine-tuned Llama on financial corpus via Ollama.
