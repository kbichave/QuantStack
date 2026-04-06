"""
IC-adjusted half-Kelly expected return estimates for MVO alpha_signals input.

Implements the Grinold-Kahn formula:
    expected_return_i = IC_i × kelly_fraction × z_i × σ_i

where σ_i is annualized volatility (daily_stdev × sqrt(252)).

Pure computation module — no database access, no LLM calls, no side effects.
Called by the risk_sizing node in trading/nodes.py (section-07).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# IC prior for strategies with < 21 days of IC history.
# Yields ~0.00075 expected return for a moderate signal at 30% vol — appropriately conservative.
IC_PRIOR: float = 0.01

# Default annualized volatility floor. Used when vol is missing or zero.
_VOL_FLOOR: float = 0.15

# ---------------------------------------------------------------------------
# Regime-conditional Kelly sizing (Gap 1)
# ---------------------------------------------------------------------------

# Keys: (regime, vol_state) where vol_state in {"normal", "high"}
REGIME_KELLY_TABLE: dict[tuple[str, str], float] = {
    ("trending_up",   "normal"): 0.50,
    ("trending_up",   "high"):   0.35,
    ("trending_down", "normal"): 0.20,
    ("trending_down", "high"):   0.20,
    ("ranging",       "normal"): 0.35,
    ("ranging",       "high"):   0.20,
    ("unknown",       "normal"): 0.15,
    ("unknown",       "high"):   0.15,
}

UNKNOWN_KELLY: float = 0.15
CONFIDENCE_FULL_REGIME_THRESHOLD: float = 0.80

# Vol state hysteresis thresholds (used in section-03, defined here for cohesion)
VOL_STATE_ENTER_THRESHOLD: float = 1.5
VOL_STATE_EXIT_THRESHOLD: float = 1.20

# Hard ceiling and floor applied AFTER all multipliers (in make_risk_sizing)
KELLY_HARD_CEILING: float = 0.50
KELLY_HARD_FLOOR: float = 0.02


TARGET_VOL: float = 0.20
VOL_SCALAR_CAP: float = 1.50


def compute_vol_state(
    current_ewma_vol: float,
    vol_63d_mean: float,
    current_state: str = "normal",
) -> str:
    """Determine volatility state using hysteresis to prevent oscillation.

    Enters "high" when current_ewma_vol > VOL_STATE_ENTER_THRESHOLD * vol_63d_mean.
    Exits "high" only when current_ewma_vol < VOL_STATE_EXIT_THRESHOLD * vol_63d_mean.
    """
    if vol_63d_mean <= 0:
        return "normal"

    ratio = current_ewma_vol / vol_63d_mean

    if current_state == "normal":
        return "high" if ratio > VOL_STATE_ENTER_THRESHOLD else "normal"

    # current_state == "high"
    return "normal" if ratio < VOL_STATE_EXIT_THRESHOLD else "high"


def regime_kelly_fraction(
    regime: str,
    vol_state: str,
    confidence: float,
) -> float:
    """Compute regime-conditional Kelly fraction.

    Returns a multiplier in [UNKNOWN_KELLY, 0.50] based on the current market
    regime, volatility state, and regime classifier confidence.

    Below CONFIDENCE_FULL_REGIME_THRESHOLD, the multiplier interpolates linearly
    toward UNKNOWN_KELLY to avoid aggressive sizing on uncertain regime signals.
    """
    key = (regime, vol_state)
    if key not in REGIME_KELLY_TABLE:
        # Try falling back to unknown regime with the given vol_state
        fallback_key = ("unknown", vol_state)
        m = REGIME_KELLY_TABLE.get(fallback_key, UNKNOWN_KELLY)
    else:
        m = REGIME_KELLY_TABLE[key]

    if confidence >= CONFIDENCE_FULL_REGIME_THRESHOLD:
        return m

    # Linear interpolation: at confidence=0 → UNKNOWN_KELLY, at threshold → m
    t = confidence / CONFIDENCE_FULL_REGIME_THRESHOLD
    return UNKNOWN_KELLY + (m - UNKNOWN_KELLY) * t


def compute_alpha_signals(
    candidates: list[dict],
    signal_ic_lookup: dict,
    volatility_lookup: dict,
    kelly_fraction: float = 0.5,
    ic_prior: float = IC_PRIOR,
) -> np.ndarray:
    """
    Computes Grinold-Kahn expected return estimates for MVO alpha_signals input.

    Args:
        candidates: list of dicts, each with keys: symbol, strategy_id, signal_value.
        signal_ic_lookup: {strategy_id: mean_rank_ic_21d}. Value is None when < 21
            days of IC observations. Missing keys are treated the same as None.
        volatility_lookup: {symbol: annualized_vol}. Must be annualized
            (daily_stdev × sqrt(252)). Missing or zero values use the 0.15 floor.
        kelly_fraction: multiplier on IC. 0.5 = half-Kelly (default).
        ic_prior: IC value used when signal_ic_lookup has None or is missing the key.

    Returns:
        1-D numpy array aligned to candidates order. Each element is the
        Grinold-Kahn expected excess return estimate for that candidate.
    """
    if not candidates:
        return np.array([])

    result = np.empty(len(candidates), dtype=float)

    for i, candidate in enumerate(candidates):
        symbol = candidate["symbol"]
        strategy_id = candidate["strategy_id"]
        z = float(candidate["signal_value"])

        # IC: use prior for None or missing
        ic_raw = signal_ic_lookup.get(strategy_id, None)
        ic = float(ic_raw) if ic_raw is not None else ic_prior

        # Volatility: use floor for missing or zero
        vol_raw = volatility_lookup.get(symbol)
        if vol_raw is None:
            logger.warning(
                "Symbol %r not found in volatility_lookup; using vol floor %.2f",
                symbol,
                _VOL_FLOOR,
            )
            vol = _VOL_FLOOR
        elif float(vol_raw) == 0.0:
            logger.warning(
                "Symbol %r has vol=0.0 in volatility_lookup; using vol floor %.2f",
                symbol,
                _VOL_FLOOR,
            )
            vol = _VOL_FLOOR
        else:
            vol = float(vol_raw)

        # Grinold-Kahn: expected_return = IC × kelly_fraction × z × σ_annualized
        result[i] = ic * kelly_fraction * z * vol

    return result
