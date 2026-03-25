# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Signal tier taxonomy — ranks indicators by institutional predictive value.

Tiers define how signals should be used in strategy construction:
  tier_1_retail     — retail noise: RSI, MACD, BB, Stoch. Exit timing ONLY.
  tier_2_smart_money — order-flow derived: FVG, CVD, VPIN, WVF. Secondary confirmation.
  tier_3_institutional — dealer/institutional positioning: GEX, LSV, insider cluster.
                         Must have ≥2 non-neutral for any entry.
  tier_4_regime_macro — context gate: HMM state, credit spreads, breadth cascade.
                         If macro is deteriorating, no bottom is reliable.

Usage
-----
    from quantstack.core.signal_tiers import SignalTier, get_signal_tier, INDICATOR_REGISTRY

    tier = get_signal_tier("rsi")          # SignalTier.RETAIL_NOISE
    is_noise = tier == SignalTier.RETAIL_NOISE
"""

from enum import Enum


class SignalTier(str, Enum):
    RETAIL_NOISE = "tier_1_retail"
    SMART_MONEY = "tier_2_smart_money"
    INSTITUTIONAL = "tier_3_institutional"
    REGIME_MACRO = "tier_4_regime_macro"


# ---------------------------------------------------------------------------
# Indicator registry
# Maps lowercase signal names → SignalTier + usage note
# ---------------------------------------------------------------------------

INDICATOR_REGISTRY: dict[str, dict] = {
    # ------------------------------------------------------------------
    # tier_1_retail — widely known, easily front-run, exit timing ONLY
    # ------------------------------------------------------------------
    "rsi": {
        "tier": SignalTier.RETAIL_NOISE,
        "use": "exit_timing",
        "note": "Retail oscillator. Valid for exit only (e.g., RSI>70 = take profit).",
    },
    "macd": {
        "tier": SignalTier.RETAIL_NOISE,
        "use": "exit_timing",
        "note": "Lagging momentum. Exit timing only.",
    },
    "macd_histogram": {
        "tier": SignalTier.RETAIL_NOISE,
        "use": "exit_timing",
        "note": "Retail noise. Exit timing only.",
    },
    "stoch_k": {
        "tier": SignalTier.RETAIL_NOISE,
        "use": "exit_timing",
        "note": "Retail oscillator. OR-logic trap in rule engine.",
    },
    "stoch_d": {
        "tier": SignalTier.RETAIL_NOISE,
        "use": "exit_timing",
        "note": "Retail oscillator. OR-logic trap in rule engine.",
    },
    "bbands": {
        "tier": SignalTier.RETAIL_NOISE,
        "use": "exit_timing",
        "note": "Retail volatility band. Use BB width (compression) as tier_2 setup filter.",
    },
    "bb_pct": {
        "tier": SignalTier.RETAIL_NOISE,
        "use": "exit_timing",
        "note": "Retail noise.",
    },
    "cci": {
        "tier": SignalTier.RETAIL_NOISE,
        "use": "exit_timing",
        "note": "Retail oscillator.",
    },
    "williams_r": {
        "tier": SignalTier.RETAIL_NOISE,
        "use": "exit_timing",
        "note": "Retail oscillator — exhaustion is better via PercentRExhaustion (tier_2).",
    },
    "adx": {
        "tier": SignalTier.RETAIL_NOISE,
        "use": "regime_filter",
        "note": "Trend strength filter — valid as context, not entry gate.",
    },
    "plus_di": {
        "tier": SignalTier.RETAIL_NOISE,
        "use": "regime_filter",
        "note": "Directional index — context only.",
    },
    "minus_di": {
        "tier": SignalTier.RETAIL_NOISE,
        "use": "regime_filter",
        "note": "Directional index — context only.",
    },
    "sma_crossover": {
        "tier": SignalTier.RETAIL_NOISE,
        "use": "exit_timing",
        "note": "Lagging retail signal. Produces OR-logic noise in rule engine.",
    },
    "sma_fast": {
        "tier": SignalTier.RETAIL_NOISE,
        "use": "context",
        "note": "Trend direction context only.",
    },
    "sma_slow": {
        "tier": SignalTier.RETAIL_NOISE,
        "use": "context",
        "note": "Trend direction context only.",
    },
    "sma_200": {
        "tier": SignalTier.RETAIL_NOISE,
        "use": "context",
        "note": "Long-term trend filter. Valid as structural context, not entry gate.",
    },
    "ema": {
        "tier": SignalTier.RETAIL_NOISE,
        "use": "context",
        "note": "Trend context only.",
    },
    "obv": {
        "tier": SignalTier.RETAIL_NOISE,
        "use": "confirmation",
        "note": "Lagging volume indicator. CVD (tier_2) is superior.",
    },
    "mfi": {
        "tier": SignalTier.RETAIL_NOISE,
        "use": "exit_timing",
        "note": "Retail money flow — less precise than CVD (tier_2).",
    },
    "candlestick_patterns": {
        "tier": SignalTier.RETAIL_NOISE,
        "use": "exit_timing",
        "note": "Retail pattern recognition. Use FVG/OB (tier_2) instead for structure.",
    },
    # ------------------------------------------------------------------
    # tier_2_smart_money — order-flow derived, not widely traded by retail
    # ------------------------------------------------------------------
    "cvd": {
        "tier": SignalTier.SMART_MONEY,
        "use": "secondary_entry",
        "note": "Cumulative Volume Delta — buy/sell pressure from candle structure.",
    },
    "cvd_divergence": {
        "tier": SignalTier.SMART_MONEY,
        "use": "secondary_entry",
        "note": "Price/CVD divergence signals hidden accumulation or distribution.",
    },
    "vpin": {
        "tier": SignalTier.SMART_MONEY,
        "use": "secondary_entry",
        "note": "Volume-Synchronized Probability of Informed Trading.",
    },
    "hawkes_intensity": {
        "tier": SignalTier.SMART_MONEY,
        "use": "secondary_entry",
        "note": "Self-exciting order flow clustering — detects institutional bursts.",
    },
    "bullish_fvg": {
        "tier": SignalTier.SMART_MONEY,
        "use": "secondary_entry",
        "note": "Bullish Fair Value Gap — unfilled institutional imbalance zone.",
    },
    "bearish_fvg": {
        "tier": SignalTier.SMART_MONEY,
        "use": "secondary_entry",
        "note": "Bearish Fair Value Gap — unfilled institutional imbalance zone.",
    },
    "bullish_ob": {
        "tier": SignalTier.SMART_MONEY,
        "use": "secondary_entry",
        "note": "Bullish Order Block — last bearish candle before institutional impulse up.",
    },
    "bearish_ob": {
        "tier": SignalTier.SMART_MONEY,
        "use": "secondary_entry",
        "note": "Bearish Order Block — last bullish candle before institutional impulse down.",
    },
    "bos_bullish": {
        "tier": SignalTier.SMART_MONEY,
        "use": "secondary_entry",
        "note": "Break of Structure bullish — higher high confirms trend change.",
    },
    "bos_bearish": {
        "tier": SignalTier.SMART_MONEY,
        "use": "secondary_entry",
        "note": "Break of Structure bearish — lower low confirms trend change.",
    },
    "choch_bullish": {
        "tier": SignalTier.SMART_MONEY,
        "use": "secondary_entry",
        "note": "Change of Character bullish — first BOS against prior trend.",
    },
    "choch_bearish": {
        "tier": SignalTier.SMART_MONEY,
        "use": "secondary_entry",
        "note": "Change of Character bearish — early warning of trend reversal.",
    },
    "wvf": {
        "tier": SignalTier.SMART_MONEY,
        "use": "secondary_entry",
        "note": "Williams VIX Fix — synthetic VIX for any instrument. Extreme = fear washout.",
    },
    "wvf_extreme": {
        "tier": SignalTier.SMART_MONEY,
        "use": "secondary_entry",
        "note": "WVF > upper Bollinger Band = extreme fear. High-conviction capitulation signal.",
    },
    "pct_r_short": {
        "tier": SignalTier.SMART_MONEY,
        "use": "secondary_entry",
        "note": "PercentR short lookback — short-term exhaustion.",
    },
    "pct_r_long": {
        "tier": SignalTier.SMART_MONEY,
        "use": "secondary_entry",
        "note": "PercentR long lookback — multi-week exhaustion.",
    },
    "exhaustion_bottom": {
        "tier": SignalTier.SMART_MONEY,
        "use": "secondary_entry",
        "note": "PercentR exhaustion bottom signal — both timeframes at extremes simultaneously.",
    },
    "kyle_lambda": {
        "tier": SignalTier.SMART_MONEY,
        "use": "secondary_entry",
        "note": "Kyle's Lambda price impact proxy — high = informed trading pressure.",
    },
    "bb_width_compression": {
        "tier": SignalTier.SMART_MONEY,
        "use": "setup_filter",
        "note": "Bollinger Band width at bottom quartile = volatility compression setup.",
    },
    "laguerre_rsi": {
        "tier": SignalTier.SMART_MONEY,
        "use": "secondary_entry",
        "note": "Laguerre RSI — reduced lag vs standard RSI.",
    },
    "supertrend": {
        "tier": SignalTier.SMART_MONEY,
        "use": "secondary_entry",
        "note": "Supertrend — better trend-following than SMA crossover.",
    },
    # ------------------------------------------------------------------
    # tier_3_institutional — dealer/institutional positioning signals
    # Primary entry gate: strategies need ≥2 non-neutral institutional signals
    # ------------------------------------------------------------------
    "opt_gex": {
        "tier": SignalTier.INSTITUTIONAL,
        "use": "primary_entry",
        "note": "Gamma Exposure — positive=dealers long gamma (mean-reversion fuel), "
                "negative=amplifying regime (avoid bottoms).",
    },
    "opt_gamma_flip": {
        "tier": SignalTier.INSTITUTIONAL,
        "use": "primary_entry",
        "note": "Strike where net GEX crosses zero. Price below flip = negative GEX regime.",
    },
    "opt_dex": {
        "tier": SignalTier.INSTITUTIONAL,
        "use": "primary_entry",
        "note": "Delta Exposure — net directional bias of open interest.",
    },
    "opt_max_pain": {
        "tier": SignalTier.INSTITUTIONAL,
        "use": "primary_entry",
        "note": "Max pain strike — expiry gravitational pull. Price converges toward it.",
    },
    "opt_iv_skew": {
        "tier": SignalTier.INSTITUTIONAL,
        "use": "primary_entry",
        "note": "OTM put IV minus OTM call IV. High skew = market paying for downside protection.",
    },
    "opt_iv_skew_zscore": {
        "tier": SignalTier.INSTITUTIONAL,
        "use": "primary_entry",
        "note": "IV skew z-score > 2.0 = maximum fear = potential sentiment extreme for longs.",
    },
    "opt_vrp": {
        "tier": SignalTier.INSTITUTIONAL,
        "use": "primary_entry",
        "note": "Volatility Risk Premium (IV - realized vol). Elevated VRP = premium selling edge.",
    },
    "opt_vanna": {
        "tier": SignalTier.INSTITUTIONAL,
        "use": "primary_entry",
        "note": "Vanna (dDelta/dVol) — dealer hedging flows when vol changes.",
    },
    "opt_charm": {
        "tier": SignalTier.INSTITUTIONAL,
        "use": "primary_entry",
        "note": "Charm (dDelta/dt) — dealer hedging flows as time decays.",
    },
    "lsv_herding": {
        "tier": SignalTier.INSTITUTIONAL,
        "use": "primary_entry",
        "note": "LSV Herding measure — institutional crowding. Positive after down-move = accumulation.",
    },
    "institutional_direction": {
        "tier": SignalTier.INSTITUTIONAL,
        "use": "primary_entry",
        "note": "Institutional ownership change direction (accumulating/stable/distributing).",
    },
    "insider_direction": {
        "tier": SignalTier.INSTITUTIONAL,
        "use": "primary_entry",
        "note": "Net insider buy/sell direction over 90 days.",
    },
    "insider_cluster_score": {
        "tier": SignalTier.INSTITUTIONAL,
        "use": "primary_entry",
        "note": "CEO/CFO-weighted insider buy ratio. >0.7 with 3+ insiders = high conviction.",
    },
    "capitulation_score": {
        "tier": SignalTier.INSTITUTIONAL,
        "use": "primary_entry",
        "note": "Composite capitulation score from get_capitulation_score tool. >0.65 = washout.",
    },
    "accumulation_score": {
        "tier": SignalTier.INSTITUTIONAL,
        "use": "primary_entry",
        "note": "Composite institutional accumulation from get_institutional_accumulation tool.",
    },
    "flow_signal": {
        "tier": SignalTier.INSTITUTIONAL,
        "use": "primary_entry",
        "note": "Composite insider + institutional flow. Weighted: 0.4×insider + 0.6×institutional.",
    },
    "put_call_oi_ratio": {
        "tier": SignalTier.INSTITUTIONAL,
        "use": "primary_entry",
        "note": "Put/call OI ratio. >1.5 = crowded short = potential squeeze fuel.",
    },
    # ------------------------------------------------------------------
    # tier_4_regime_macro — context gates, must not be deteriorating for entries
    # ------------------------------------------------------------------
    "hmm_regime": {
        "tier": SignalTier.REGIME_MACRO,
        "use": "context_gate",
        "note": "HMM 4-state regime (LOW_VOL_BULL/BEAR, HIGH_VOL_BULL/BEAR). "
                "Probability-based, not binary.",
    },
    "hmm_stability": {
        "tier": SignalTier.REGIME_MACRO,
        "use": "context_gate",
        "note": "HMM state stability score. >0.7 = regime confirmed, <0.5 = transition.",
    },
    "regime": {
        "tier": SignalTier.REGIME_MACRO,
        "use": "context_gate",
        "note": "Simplified regime label (trending_up/down/ranging). Use hmm_stability to weight.",
    },
    "credit_regime": {
        "tier": SignalTier.REGIME_MACRO,
        "use": "context_gate",
        "note": "Credit market regime from get_credit_market_signals. "
                "If widening, no bottom is reliable.",
    },
    "hy_spread_zscore": {
        "tier": SignalTier.REGIME_MACRO,
        "use": "context_gate",
        "note": "HY-IG spread z-score. Rising = risk-off, spreads widening.",
    },
    "breadth_score": {
        "tier": SignalTier.REGIME_MACRO,
        "use": "context_gate",
        "note": "% of sector ETFs above 50d SMA. <0.3 = broad market washout.",
    },
    "breadth_divergence": {
        "tier": SignalTier.REGIME_MACRO,
        "use": "context_gate",
        "note": "SPY at new low but breadth score stabilizing = hidden accumulation signal.",
    },
    "yield_curve_slope": {
        "tier": SignalTier.REGIME_MACRO,
        "use": "context_gate",
        "note": "TLT/SHY ratio proxy. Steepening into selloff = bottoms more reliable.",
    },
    "macro_rate_regime": {
        "tier": SignalTier.REGIME_MACRO,
        "use": "context_gate",
        "note": "Rate cycle position — rising rates hurt growth, stable rates enable bottoms.",
    },
    "egarch_regime": {
        "tier": SignalTier.REGIME_MACRO,
        "use": "context_gate",
        "note": "EGARCH persistence >1.0 = explosive vol regime — widen stops, reduce size.",
    },
    "piotroski_f_score": {
        "tier": SignalTier.REGIME_MACRO,
        "use": "context_gate",
        "note": "Quality gate for investment strategies. ≥7 = strong fundamentals.",
    },
}


def get_signal_tier(signal_name: str) -> SignalTier | None:
    """Return the tier for a signal name. Case-insensitive. Returns None if unknown."""
    entry = INDICATOR_REGISTRY.get(signal_name.lower())
    return entry["tier"] if entry else None


def get_signals_by_tier(tier: SignalTier) -> list[str]:
    """Return all signal names for a given tier."""
    return [name for name, info in INDICATOR_REGISTRY.items() if info["tier"] == tier]


SIGNAL_HIERARCHY_PROMPT = """
## SIGNAL HIERARCHY — Mandatory for all strategies

Signals are ranked by institutional predictive value. Strategies MUST satisfy
minimum tier requirements before registration.

| Tier | Name | Role | Min per Strategy | Examples |
|------|------|------|-----------------|---------|
| tier_3_institutional | Institutional | PRIMARY entry gate | ≥ 2 non-neutral | GEX, LSV herding, insider_cluster, opt_iv_skew_zscore, capitulation_score |
| tier_2_smart_money | Smart Money | SECONDARY confirmation | ≥ 1 | FVG, CVD divergence, WVF extreme, Order Blocks, exhaustion_bottom |
| tier_4_regime_macro | Regime/Macro | CONTEXT gate | ≥ 1 non-deteriorating | HMM state, credit_regime, breadth_score, piotroski_f_score |
| tier_1_retail | Retail Noise | EXIT timing ONLY | 0 as entry gate | RSI, MACD, Bollinger Bands, Stochastic, SMA crossover |

### HARD RULES
- **NEVER** use tier_1_retail as a primary entry condition — they are retail noise,
  widely front-run by market makers, and produce OR-logic traps in the MCP rule engine
  (workshop_lessons.md, iteration 3: RSI/Stoch gates fire ~90% of the time = always-on signal)
- **ALWAYS** check tier_4_regime_macro context first: if credit_regime == "widening" OR
  breadth_score < 0.15, no bottom strategy should enter regardless of tier_3 signals
- **ATR** is tier_1 as an entry signal but tier_3 as a risk-sizing input — use it for stops/sizing
- **BB width compression** is a valid tier_2 SETUP FILTER but the entry itself needs tier_3 confirmation
- **Retail noise tools useful for**: exit timing (RSI>70 = take profit), rough trend context

### NEW TOOLS FOR BOTTOM DETECTION
- `get_capitulation_score(symbol)` — composite wash-out score (tier_3)
- `get_institutional_accumulation(symbol)` — GEX + insider cluster + IV skew extreme (tier_3)
- `get_credit_market_signals()` — HYG/LQD ratio, yield curve, dollar/gold (tier_4)
- `get_market_breadth()` — sector ETF breadth cascade (tier_4)
"""
