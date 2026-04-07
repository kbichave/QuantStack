"""
Failure mode taxonomy for losing trades.

Classifies trade failures into categories to enable focused research
and systematic strategy improvement.
"""

from enum import Enum
from typing import Optional


class FailureMode(str, Enum):
    """
    Taxonomy of trade failure modes, prioritized by diagnostic value.

    Categories ordered by detection priority:
    1. REGIME_MISMATCH - strategy deployed in wrong market regime
    2. FACTOR_CROWDING - signal degradation due to crowded positioning
    3. DATA_STALE - stale/missing data led to bad decision
    4. TIMING_ERROR - correct thesis, wrong entry/exit timing
    5. THESIS_WRONG - fundamental thesis was incorrect
    6. BLACK_SWAN - tail event beyond normal strategy risk
    7. LIQUIDITY_TRAP - excessive slippage ate into expected edge
    8. MODEL_DEGRADATION - model's feature distribution shifted (PSI > 0.25)
    9. SIGNAL_DECAY - information coefficient decayed below useful threshold
    10. ADVERSE_SELECTION - loss within minutes of entry (toxic flow)
    11. CORRELATION_BREAKDOWN - cross-asset correlation diverged > 2 sigma
    12. UNCLASSIFIED - insufficient data to classify
    """
    REGIME_MISMATCH = "regime_mismatch"
    FACTOR_CROWDING = "factor_crowding"
    DATA_STALE = "data_stale"
    TIMING_ERROR = "timing_error"
    THESIS_WRONG = "thesis_wrong"
    BLACK_SWAN = "black_swan"
    LIQUIDITY_TRAP = "liquidity_trap"
    MODEL_DEGRADATION = "model_degradation"
    SIGNAL_DECAY = "signal_decay"
    ADVERSE_SELECTION = "adverse_selection"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    UNCLASSIFIED = "unclassified"


def classify_failure(
    realized_pnl_pct: float,
    regime_at_entry: str,
    regime_at_exit: str,
    strategy_id: str,
    symbol: str,
    entry_price: float,
    exit_price: float,
    data_freshness: Optional[float] = None,
    key_levels: Optional[list[float]] = None,
    historical_losses: Optional[list[float]] = None,
    slippage_pct: Optional[float] = None,
    psi_at_entry: Optional[float] = None,
    ic_trailing_10d: Optional[float] = None,
    holding_minutes: Optional[float] = None,
    correlation_z_score: Optional[float] = None,
) -> FailureMode:
    """
    Classify a losing trade into a failure mode using rule-based heuristics.

    Rules applied in priority order:
    1. REGIME_MISMATCH if regime_at_entry != regime_at_exit
    2. DATA_STALE if data_freshness > 60 minutes (for intraday)
    3. BLACK_SWAN if loss > 3 std from historical loss distribution
    4. LIQUIDITY_TRAP if slippage_pct > 0.02 (2%)
    5. MODEL_DEGRADATION if psi_at_entry > 0.25
    6. SIGNAL_DECAY if ic_trailing_10d < 0.005
    7. ADVERSE_SELECTION if loss within 30 min of entry
    8. CORRELATION_BREAKDOWN if correlation_z_score > 2.0
    9. TIMING_ERROR if entry within 0.5% of any key level
    10. UNCLASSIFIED if no rule matches

    Args:
        realized_pnl_pct: Realized P&L as percentage (negative for loss)
        regime_at_entry: Market regime when position opened
        regime_at_exit: Market regime when position closed
        strategy_id: Strategy identifier
        symbol: Trading symbol
        entry_price: Entry price
        exit_price: Exit price
        data_freshness: Minutes since last data update (optional)
        key_levels: List of key price levels (support/resistance)
        historical_losses: List of historical loss percentages for this strategy
        slippage_pct: Execution slippage as a fraction (0.02 = 2%)
        psi_at_entry: Population Stability Index at entry time
        ic_trailing_10d: Information coefficient over trailing 10 days
        holding_minutes: Minutes between entry and exit
        correlation_z_score: Z-score of cross-asset correlation divergence

    Returns:
        FailureMode enum value
    """
    # Rule 1: Regime mismatch
    if regime_at_entry != regime_at_exit:
        return FailureMode.REGIME_MISMATCH

    # Rule 2: Data staleness
    if data_freshness is not None and data_freshness > 60:
        return FailureMode.DATA_STALE

    # Rule 3: Black swan (requires sufficient history)
    if historical_losses is not None and len(historical_losses) >= 20:
        import statistics
        try:
            mean_loss = statistics.mean(historical_losses)
            std_loss = statistics.stdev(historical_losses)
            # Loss is more negative than 3 std below mean
            if realized_pnl_pct < (mean_loss - 3 * std_loss):
                return FailureMode.BLACK_SWAN
        except statistics.StatisticsError:
            # Not enough variance or other statistical issue
            pass

    # Rule 4: Liquidity trap (excessive slippage)
    if slippage_pct is not None and slippage_pct > 0.02:
        return FailureMode.LIQUIDITY_TRAP

    # Rule 5: Model degradation (feature distribution shift)
    if psi_at_entry is not None and psi_at_entry > 0.25:
        return FailureMode.MODEL_DEGRADATION

    # Rule 6: Signal decay (information coefficient below useful threshold)
    if ic_trailing_10d is not None and ic_trailing_10d < 0.005:
        return FailureMode.SIGNAL_DECAY

    # Rule 7: Adverse selection (loss within 30 minutes of entry)
    if holding_minutes is not None and holding_minutes <= 30 and realized_pnl_pct < 0:
        return FailureMode.ADVERSE_SELECTION

    # Rule 8: Correlation breakdown (cross-asset divergence > 2 sigma)
    if correlation_z_score is not None and abs(correlation_z_score) > 2.0:
        return FailureMode.CORRELATION_BREAKDOWN

    # Rule 9: Timing error (entry near key level)
    if key_levels is not None and len(key_levels) > 0:
        for level in key_levels:
            pct_distance = abs(entry_price - level) / entry_price
            if pct_distance <= 0.005:  # Within 0.5%
                return FailureMode.TIMING_ERROR

    # Rule 10: No rule matched
    return FailureMode.UNCLASSIFIED


def compute_research_priority(
    cumulative_loss_30d: float,
    days_since_last_loss: int,
) -> int:
    """
    Compute research queue priority for a failing strategy.

    Priority formula: min(9, int(cumulative_loss_30d * recency_weight * 10))
    where recency_weight = 0.95^days_since_last_loss

    Args:
        cumulative_loss_30d: Total cumulative loss over last 30 days (positive value)
        days_since_last_loss: Days since most recent loss occurred

    Returns:
        Priority score 0-9 (higher = more urgent)
    """
    recency_weight = 0.95 ** days_since_last_loss
    priority = int(cumulative_loss_30d * recency_weight * 10)
    return min(9, priority)
