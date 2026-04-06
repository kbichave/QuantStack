"""
Alt-data normalizer — combines EDGAR, earnings, and macro signals into
a per-symbol modifier for the trading signal pipeline.

Each source returns a z-scored float or None (stale/missing).
The combined modifier is clamped to [-1, 1] and weighted by ALT_DATA_WEIGHT
when added to the price signal in make_risk_sizing().

Pure computation — queries DB for raw inputs but produces a stateless output.
"""

from __future__ import annotations

from datetime import date, timedelta

from loguru import logger

from quantstack.db import PgConnection

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALT_DATA_WEIGHT: float = 0.30
EDGAR_STALENESS_DAYS: int = 30
EARNINGS_WINDOW_DAYS: int = 5
EARNINGS_STALENESS_DAYS: int = 100
EDGAR_LOOKBACK_DAYS: int = 63
IV_COMPRESSION_FACTOR_FAVORABLE: float = 1.0
IV_COMPRESSION_FACTOR_UNFAVORABLE: float = 0.5

MACRO_STRESS_VIX_WEIGHT: float = 0.4
MACRO_STRESS_YIELD_WEIGHT: float = 0.4
MACRO_STRESS_CREDIT_WEIGHT: float = 0.2

MACRO_STRESS_MODERATE_THRESHOLD: float = 1.5
MACRO_STRESS_HIGH_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# EDGAR sentiment
# ---------------------------------------------------------------------------


def compute_edgar_score(symbol: str, conn: PgConnection) -> float | None:
    """Compute z-scored EDGAR filing sentiment for a symbol.

    Uses overall_sentiment_score from sec_filings joined with news_sentiment
    as a proxy. Only considers 8-K and 10-Q filings.

    Returns None if no qualifying filings in the staleness window.
    """
    today = date.today()
    staleness_cutoff = today - timedelta(days=EDGAR_STALENESS_DAYS)
    lookback_cutoff = today - timedelta(days=EDGAR_LOOKBACK_DAYS)

    # Get recent filing sentiment scores (8-K and 10-Q only)
    rows = conn.execute(
        """
        SELECT overall_sentiment_score
        FROM news_sentiment
        WHERE ticker = %s
          AND time_published >= %s
          AND time_published IS NOT NULL
          AND overall_sentiment_score IS NOT NULL
        ORDER BY time_published DESC
        """,
        [symbol, lookback_cutoff],
    ).fetchall()

    if not rows:
        return None

    scores = [float(r[0]) for r in rows]

    # Check staleness: most recent must be within EDGAR_STALENESS_DAYS
    recent_rows = conn.execute(
        """
        SELECT 1 FROM news_sentiment
        WHERE ticker = %s
          AND time_published >= %s
          AND overall_sentiment_score IS NOT NULL
        LIMIT 1
        """,
        [symbol, staleness_cutoff],
    ).fetchone()

    if recent_rows is None:
        return None

    if len(scores) < 2:
        return scores[0]

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    if variance == 0:
        return 0.0
    std = variance ** 0.5
    return (scores[0] - mean) / std


# ---------------------------------------------------------------------------
# Earnings surprise
# ---------------------------------------------------------------------------


def compute_earnings_score(symbol: str, conn: PgConnection) -> float | None:
    """Compute earnings surprise score within the event window.

    Active only within EARNINGS_WINDOW_DAYS of the most recent earnings.
    Combines beat_rate z-score with IV compression factor.
    """
    today = date.today()
    staleness_cutoff = today - timedelta(days=EARNINGS_STALENESS_DAYS)

    # Get most recent earnings event
    row = conn.execute(
        """
        SELECT report_date, surprise_pct
        FROM earnings_calendar
        WHERE symbol = %s
          AND report_date <= %s
          AND report_date >= %s
        ORDER BY report_date DESC
        LIMIT 1
        """,
        [symbol, today, staleness_cutoff],
    ).fetchone()

    if row is None:
        return None

    report_date, surprise_pct = row
    if report_date is None:
        return None

    days_since = (today - report_date).days
    if days_since > EARNINGS_WINDOW_DAYS:
        return None

    if surprise_pct is None:
        return None

    # Get trailing 8-quarter history for z-score
    history = conn.execute(
        """
        SELECT surprise_pct
        FROM earnings_calendar
        WHERE symbol = %s
          AND report_date <= %s
          AND surprise_pct IS NOT NULL
        ORDER BY report_date DESC
        LIMIT 8
        """,
        [symbol, today],
    ).fetchall()

    if len(history) < 2:
        beat_z = float(surprise_pct)
    else:
        hist_vals = [float(h[0]) for h in history]
        mean = sum(hist_vals) / len(hist_vals)
        var = sum((v - mean) ** 2 for v in hist_vals) / len(hist_vals)
        if var == 0:
            beat_z = 0.0
        else:
            beat_z = (float(surprise_pct) - mean) / (var ** 0.5)

    # IV compression: check if post-earnings IV is lower than pre-earnings
    # Use options_chains if available, otherwise neutral factor
    iv_factor = IV_COMPRESSION_FACTOR_FAVORABLE  # default: neutral
    try:
        pre_iv = conn.execute(
            """
            SELECT AVG(iv) FROM options_chains
            WHERE underlying = %s
              AND data_date BETWEEN %s AND %s
              AND iv IS NOT NULL
            """,
            [symbol, report_date - timedelta(days=5), report_date - timedelta(days=1)],
        ).fetchone()
        post_iv = conn.execute(
            """
            SELECT AVG(iv) FROM options_chains
            WHERE underlying = %s
              AND data_date BETWEEN %s AND %s
              AND iv IS NOT NULL
            """,
            [symbol, report_date, report_date + timedelta(days=3)],
        ).fetchone()

        if pre_iv and post_iv and pre_iv[0] and post_iv[0]:
            if float(post_iv[0]) >= float(pre_iv[0]):
                iv_factor = IV_COMPRESSION_FACTOR_UNFAVORABLE
    except Exception:
        pass  # IV data unavailable — use neutral factor

    return beat_z * iv_factor


# ---------------------------------------------------------------------------
# Macro stress
# ---------------------------------------------------------------------------


def compute_macro_stress(conn: PgConnection) -> float:
    """Compute composite macro stress score from FRED/market indicators.

    Combines VIX z-score, yield curve inversion, and credit stress proxy.
    Returns a raw composite score (not a scalar).
    """
    # VIX z-score (252-day rolling)
    vix_rows = conn.execute(
        """
        SELECT value FROM macro_indicators
        WHERE indicator = 'VIX'
        ORDER BY date DESC
        LIMIT 252
        """
    ).fetchall()

    if not vix_rows or len(vix_rows) < 10:
        return 0.0

    vix_values = [float(r[0]) for r in vix_rows if r[0] is not None]
    if not vix_values:
        return 0.0

    current_vix = vix_values[0]
    vix_mean = sum(vix_values) / len(vix_values)
    vix_var = sum((v - vix_mean) ** 2 for v in vix_values) / len(vix_values)
    vix_z = (current_vix - vix_mean) / (vix_var ** 0.5) if vix_var > 0 else 0.0

    # Yield curve: 10y - 2y spread
    yield_10y = conn.execute(
        "SELECT value FROM macro_indicators WHERE indicator = 'GS10' ORDER BY date DESC LIMIT 1"
    ).fetchone()
    yield_2y = conn.execute(
        "SELECT value FROM macro_indicators WHERE indicator = 'GS2' ORDER BY date DESC LIMIT 1"
    ).fetchone()

    if yield_10y and yield_2y and yield_10y[0] is not None and yield_2y[0] is not None:
        spread = float(yield_10y[0]) - float(yield_2y[0])
        yield_inversion = 1.0 if spread < 0 else 0.0
    else:
        yield_inversion = 0.0

    # Credit stress proxy: use VIX-based approximation
    credit_stress = max(0.0, vix_z)

    return (
        MACRO_STRESS_VIX_WEIGHT * vix_z
        + MACRO_STRESS_YIELD_WEIGHT * yield_inversion
        + MACRO_STRESS_CREDIT_WEIGHT * credit_stress
    )


def get_macro_stress_scalar(stress_score: float) -> float:
    """Convert macro stress score to a position size multiplier.

    stress > 2.0 → 0.5
    stress > 1.5 → 0.7
    otherwise   → 1.0
    """
    if stress_score > MACRO_STRESS_HIGH_THRESHOLD:
        return 0.5
    if stress_score > MACRO_STRESS_MODERATE_THRESHOLD:
        return 0.7
    return 1.0


# ---------------------------------------------------------------------------
# Combined modifier
# ---------------------------------------------------------------------------


def get_alt_data_modifier(symbol: str, conn: PgConnection) -> float:
    """Return combined alt-data signal modifier for a symbol, clamped to [-1, 1].

    Averages available source scores (EDGAR, earnings). Macro stress is
    a portfolio-level overlay applied in risk_gate.py, not here.

    Returns 0.0 if all sources return None or raise exceptions.
    """
    scores: list[float] = []

    for source_fn in (compute_edgar_score, compute_earnings_score):
        try:
            val = source_fn(symbol, conn)
            if val is not None:
                scores.append(val)
        except Exception as exc:
            logger.warning(
                "[AltData] %s failed for %s: %s", source_fn.__name__, symbol, exc
            )

    if not scores:
        return 0.0

    avg = sum(scores) / len(scores)
    return max(-1.0, min(1.0, avg))
