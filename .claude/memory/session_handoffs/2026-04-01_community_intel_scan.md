# Community Intel Scan — 2026-04-01

**Agent:** community-intel  
**Scan Date:** 2026-04-01  
**Sources Scanned:** 8 major sources (arXiv, Medium, GitHub)  
**Discoveries:** 12 unique, filtered: 1 PASS + 4 INVESTIGATE + 5 SKIP + 2 TOOLS

---

## Raw Discoveries (Phase 1)

| # | Title | Source | Date | Evidence | Status |
|----|-------|--------|------|----------|--------|
| 1 | Deep Learning Enhanced Multi-Day Turnover Algorithm | arXiv | 2025-06 | Peer-reviewed + live backtest (Sharpe 1.87) | **QUEUED** |
| 2 | Be Water: Evolutionary Proof for Trend-Following | arXiv | 2026-04-01 | Peer-reviewed preprint | INVESTIGATE |
| 3 | Why Kalman Filter Beats Moving Averages | Medium | 2026-03 | Practitioner tutorial | INVESTIGATE |
| 4 | Monotone 2D Integration for Mean-CVaR Portfolio Opt | arXiv | 2026-03-26 | Peer-reviewed preprint | INVESTIGATE |
| 5 | Deep Learning Hedging with No-Transaction Bands | arXiv | 2026-04-01 | Peer-reviewed preprint | INVESTIGATE |
| 6 | Hedging with Sparse Reward RL + Attention | arXiv | 2025-03 | Peer-reviewed preprint | SKIP (no metrics) |
| 7 | TLOB: Transformer for LOB Prediction | arXiv | 2025-02 | Peer-reviewed + SoTA | SKIP (LOB data unavailable) |
| 8 | Nonlinear Factor Decomposition via KAN | arXiv | 2026-03-31 | Peer-reviewed preprint | SKIP (no backtest) |
| 9 | Ultra-short-term Volatility Surfaces | arXiv | 2026-04-01 | Peer-reviewed preprint | SKIP (requires tick data) |
| 10 | Skew-Enhanced SABR (Chinese Options) | arXiv | 2026-03-31 | Peer-reviewed preprint | SKIP (China market) |
| 11 | Model Predictive Control for Trade Execution | arXiv | 2026-04-01 | Peer-reviewed preprint | SKIP (execution layer) |
| 12 | Option Pricing on AMM Tokens | arXiv | 2026-04-01 | Peer-reviewed preprint | SKIP (DeFi/crypto only) |

---

## Queued for Research (1 strategy)

### 1. Deep Learning Enhanced Multi-Day Turnover Algorithm
**Priority:** 7 (high)  
**Source:** arXiv 2506.06356 (Yimin Du, June 2025)  
**Evidence Type:** peer_reviewed + live backtest  
**Empirical Results (2021-2024 OOS):** 15.2% annualized, Sharpe 1.87, max DD <5%

**Mechanism:** Deep cross-sectional neural networks + mixture models + vol-based sizing  

**Implementation:** Extend alpha_discovery/engine.py for neural rule generation on SPY/QQQ/XLK components. Data: daily OHLCV (Alpha Vantage).

**Gap vs Current:** Current regime_momentum is rule-based OR-logic (0.79 IS / 1.346 OOS). New: learned cross-sectional prediction with automated feature discovery.

**Next Steps:** H_DEEPML_001_CrossSectionalDL hypothesis. Backtest 2020-2024 on multi-symbol. Target Sharpe > 1.346.

---

## Investigated (4 items)

1. **Be Water (Evolutionary Validation)** — Archive for regime confirmation  
2. **Kalman Filter for Signal Preprocessing** — Consider for signal_generator.py  
3. **Mean-CVaR Portfolio Optimization** — Archive for multi-asset allocation phase  
4. **Deep Learning Hedging** — Archive for options strategy improvement  

---

## Skipped (5 items)

| Title | Reason |
|-------|--------|
| Sparse Reward RL Hedging | No backtest results |
| TLOB Transformer | LOB data unavailable |
| KAN Factor Decomposition | Theory-only |
| Ultra-short Volatility | Requires tick data |
| Skew-Enhanced SABR | China market |

---

## Tools Discovered

1. **TradingAgents** (GitHub, 45K stars) — Multi-agent LLM orchestration framework  
2. **ai-hedge-fund** (GitHub, 49K stars) — LLM-based hedge fund automation  

---

## Summary

- Total scanned: 12
- Queued: 1
- Investigated: 4
- Skipped: 5
- Tools: 2
- Data constraints: 50% need LOB/tick data we don't have

**Scan conducted:** 2026-04-01 13:07 UTC  
**Queued in research_queue:** 1 strategy  
**Next scan:** 2026-05-01
