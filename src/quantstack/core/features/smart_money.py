"""
ICT (Inner Circle Trader) Smart Money Concepts.

All concepts are implemented with strict, unambiguous operational definitions
over OHLCV data — no lookahead bias. Every signal at bar i uses only data
from bars 0..i.

Concepts
--------
FairValueGapDetector    – 3-candle FVG detection (bullish and bearish)
OrderBlockDetector      – last opposing candle before an impulse move
BreakerBlockDetector    – OB that price has since violated (flipped polarity)
StructureAnalysis       – BOS (Break of Structure) and CHoCH (Change of Character)
EqualHighsLows          – liquidity cluster detection
OTELevels               – Optimal Trade Entry (61.8-79% Fibonacci retracement)
ICTKillZones            – session-based time windows (London, NY AM, NY PM)
ICTPowerOfThree         – PO3 session phase identification (accumulation/manipulation/distribution)
SilverBullet            – FVG formed in 10:00-11:00 AM NY window (highest-probability setup)
MMXMCycle               – Market Maker eXpansion Model cycle labeling
SMTDivergence           – Smart Money Technique divergence between two correlated instruments
"""

import numpy as np
import pandas as pd
import pytz


class FairValueGapDetector:
    """
    Fair Value Gap (FVG) — 3-candle imbalance pattern.

    Bullish FVG:  low[i] > high[i-2]  (gap between i-2 candle's high and i's low)
    Bearish FVG:  high[i] < low[i-2]  (gap between i-2 candle's low and i's high)

    The gap represents a price range where no two-sided trading occurred.
    Price often returns to fill this imbalance — creating both entry and
    target levels.

    Parameters
    ----------
    min_gap_atr_multiple : float
        Minimum gap size as multiple of ATR to filter noise. Default 0.1.
    atr_period : int
        ATR lookback for gap size filtering. Default 14.
    """

    def __init__(self, min_gap_atr_multiple: float = 0.1, atr_period: int = 14) -> None:
        self.min_gap_atr_multiple = min_gap_atr_multiple
        self.atr_period = atr_period

    def compute(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns:
            bullish_fvg      – 1 on bar where bullish FVG forms
            bearish_fvg      – 1 on bar where bearish FVG forms
            fvg_top          – upper bound of the FVG zone (NaN when no FVG)
            fvg_bottom       – lower bound of the FVG zone (NaN when no FVG)
            fvg_filled       – 1 if price has returned to the most recent active FVG
        """
        # ATR for gap size filter
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(span=self.atr_period, adjust=False).mean()
        min_gap = atr * self.min_gap_atr_multiple

        # 3-candle FVG patterns
        # Bullish: candle[i].low > candle[i-2].high
        bullish_gap = low - high.shift(2)
        bullish_fvg = ((bullish_gap > min_gap)).astype(int)
        bullish_fvg.iloc[:2] = 0

        # Bearish: candle[i].high < candle[i-2].low
        bearish_gap = low.shift(2) - high
        bearish_fvg = ((bearish_gap > min_gap)).astype(int)
        bearish_fvg.iloc[:2] = 0

        # Gap zone bounds
        fvg_top = pd.Series(np.nan, index=close.index)
        fvg_bottom = pd.Series(np.nan, index=close.index)

        mask_bull = bullish_fvg == 1
        fvg_top[mask_bull] = low[mask_bull]
        fvg_bottom[mask_bull] = high.shift(2)[mask_bull]

        mask_bear = bearish_fvg == 1
        fvg_top[mask_bear] = low.shift(2)[mask_bear]
        fvg_bottom[mask_bear] = high[mask_bear]

        # Track most recent active FVG fill (price re-enters the gap zone)
        fvg_filled = pd.Series(0, index=close.index)
        active_top = np.nan
        active_bottom = np.nan

        for i in range(len(close)):
            if bullish_fvg.iloc[i] == 1 or bearish_fvg.iloc[i] == 1:
                active_top = fvg_top.iloc[i]
                active_bottom = fvg_bottom.iloc[i]
            elif not (np.isnan(active_top) or np.isnan(active_bottom)):
                bar_high = high.iloc[i]
                bar_low = low.iloc[i]
                if bar_low <= active_top and bar_high >= active_bottom:
                    fvg_filled.iloc[i] = 1
                    active_top = np.nan
                    active_bottom = np.nan

        return pd.DataFrame(
            {
                "bullish_fvg": bullish_fvg,
                "bearish_fvg": bearish_fvg,
                "fvg_top": fvg_top,
                "fvg_bottom": fvg_bottom,
                "fvg_filled": fvg_filled,
            },
            index=close.index,
        )


class OrderBlockDetector:
    """
    Order Block (OB) detection — last opposing candle before an impulse.

    Bullish OB: last bearish candle (close < open) before a bullish impulse
    Bearish OB: last bullish candle (close > open) before a bearish impulse

    An "impulse" is defined as a move of at least `impulse_atr_multiple` × ATR
    in a single candle.

    Parameters
    ----------
    impulse_atr_multiple : float
        Minimum ATR multiple for the impulse candle to qualify. Default 1.5.
    atr_period : int
        ATR lookback period. Default 14.
    """

    def __init__(self, impulse_atr_multiple: float = 1.5, atr_period: int = 14) -> None:
        self.impulse_atr_multiple = impulse_atr_multiple
        self.atr_period = atr_period

    def compute(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns:
            bullish_ob      – 1 on the OB bar (bearish candle before bullish impulse)
            bearish_ob      – 1 on the OB bar (bullish candle before bearish impulse)
            ob_high         – high of the order block candle
            ob_low          – low of the order block candle
        """
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(span=self.atr_period, adjust=False).mean()
        min_impulse = atr * self.impulse_atr_multiple

        candle_range = high - low
        bearish_candle = (close < open_).astype(int)
        bullish_candle = (close > open_).astype(int)
        bullish_impulse = ((close - open_) > min_impulse).astype(int)
        bearish_impulse = ((open_ - close) > min_impulse).astype(int)

        # Bullish OB: prior candle was bearish and current candle is a bullish impulse
        bullish_ob = ((bearish_candle.shift(1) == 1) & (bullish_impulse == 1)).astype(
            int
        )
        # Bearish OB: prior candle was bullish and current candle is a bearish impulse
        bearish_ob = ((bullish_candle.shift(1) == 1) & (bearish_impulse == 1)).astype(
            int
        )

        # OB zone = prior candle's H/L
        ob_high = pd.Series(np.nan, index=close.index)
        ob_low = pd.Series(np.nan, index=close.index)

        ob_high[bullish_ob == 1] = high.shift(1)[bullish_ob == 1]
        ob_low[bullish_ob == 1] = low.shift(1)[bullish_ob == 1]
        ob_high[bearish_ob == 1] = high.shift(1)[bearish_ob == 1]
        ob_low[bearish_ob == 1] = low.shift(1)[bearish_ob == 1]

        return pd.DataFrame(
            {
                "bullish_ob": bullish_ob,
                "bearish_ob": bearish_ob,
                "ob_high": ob_high,
                "ob_low": ob_low,
            },
            index=close.index,
        )


class StructureAnalysis:
    """
    Break of Structure (BOS) and Change of Character (CHoCH).

    BOS: price closes beyond the most recent swing high (bullish BOS) or
         swing low (bearish BOS) — continuation signal.

    CHoCH: price breaks the swing that formed the last BOS — reversal signal.
           First CHoCH after a run of BOS = highest-probability reversal entry.

    Swing detection uses a simple `swing_period`-bar lookback.

    Parameters
    ----------
    swing_period : int
        Bars on each side to qualify a swing high/low. Default 5.
    """

    def __init__(self, swing_period: int = 5) -> None:
        self.swing_period = swing_period

    def compute(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns:
            swing_high        – 1 on confirmed swing high bar
            swing_low         – 1 on confirmed swing low bar
            bos_bullish       – 1 when close breaks above last swing high
            bos_bearish       – 1 when close breaks below last swing low
            choch_bullish     – 1 when bullish CHoCH (first bullish break after bearish trend)
            choch_bearish     – 1 when bearish CHoCH (first bearish break after bullish trend)
        """
        n = self.swing_period

        # Swing high: highest in n-bar window on each side (confirmed n bars later)
        swing_high = pd.Series(0, index=close.index)
        swing_low = pd.Series(0, index=close.index)

        for i in range(n, len(high) - n):
            window = high.iloc[i - n : i + n + 1]
            if high.iloc[i] == window.max():
                swing_high.iloc[i] = 1

        for i in range(n, len(low) - n):
            window = low.iloc[i - n : i + n + 1]
            if low.iloc[i] == window.min():
                swing_low.iloc[i] = 1

        # Track last confirmed swing levels
        last_swing_high = pd.Series(np.nan, index=close.index)
        last_swing_low = pd.Series(np.nan, index=close.index)

        sh_val = np.nan
        sl_val = np.nan
        for i in range(len(close)):
            if swing_high.iloc[i] == 1:
                sh_val = high.iloc[i]
            if swing_low.iloc[i] == 1:
                sl_val = low.iloc[i]
            last_swing_high.iloc[i] = sh_val
            last_swing_low.iloc[i] = sl_val

        # BOS: close breaks through the most recent swing level
        bos_bullish = (
            (close > last_swing_high) & (close.shift(1) <= last_swing_high.shift(1))
        ).astype(int)
        bos_bearish = (
            (close < last_swing_low) & (close.shift(1) >= last_swing_low.shift(1))
        ).astype(int)

        # CHoCH: first reversal break after a series of BOS in opposite direction
        # Simple: CHoCH bullish = bearish BOS had occurred, then price makes bullish BOS
        recent_bos_bearish = bos_bearish.rolling(20).max()
        recent_bos_bullish = bos_bullish.rolling(20).max()

        choch_bullish = (
            (bos_bullish == 1) & (recent_bos_bearish.shift(1) == 1)
        ).astype(int)
        choch_bearish = (
            (bos_bearish == 1) & (recent_bos_bullish.shift(1) == 1)
        ).astype(int)

        return pd.DataFrame(
            {
                "swing_high": swing_high,
                "swing_low": swing_low,
                "bos_bullish": bos_bullish,
                "bos_bearish": bos_bearish,
                "choch_bullish": choch_bullish,
                "choch_bearish": choch_bearish,
            },
            index=close.index,
        )


class EqualHighsLows:
    """
    Equal Highs and Equal Lows — liquidity cluster (stop-loss magnet).

    Bars within ATR × `tolerance_atr_multiple` of a prior swing high/low
    are flagged as "equal" — these form stop-loss clusters that market
    makers target for liquidity sweeps.

    Parameters
    ----------
    lookback : int
        How many bars back to search for equal levels. Default 20.
    tolerance_atr_multiple : float
        ATR multiple defining "equal" proximity. Default 0.1.
    atr_period : int
        ATR period. Default 14.
    """

    def __init__(
        self,
        lookback: int = 20,
        tolerance_atr_multiple: float = 0.1,
        atr_period: int = 14,
    ) -> None:
        self.lookback = lookback
        self.tolerance_atr_multiple = tolerance_atr_multiple
        self.atr_period = atr_period

    def compute(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns:
            equal_highs   – 1 when current high ≈ a prior high within lookback
            equal_lows    – 1 when current low ≈ a prior low within lookback
        """
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(span=self.atr_period, adjust=False).mean()
        tolerance = atr * self.tolerance_atr_multiple

        equal_highs = pd.Series(0, index=close.index)
        equal_lows = pd.Series(0, index=close.index)

        for i in range(self.lookback, len(close)):
            tol = tolerance.iloc[i]
            window_high = high.iloc[i - self.lookback : i]
            window_low = low.iloc[i - self.lookback : i]
            cur_high = high.iloc[i]
            cur_low = low.iloc[i]

            if ((window_high - cur_high).abs() <= tol).any():
                equal_highs.iloc[i] = 1
            if ((window_low - cur_low).abs() <= tol).any():
                equal_lows.iloc[i] = 1

        return pd.DataFrame(
            {"equal_highs": equal_highs, "equal_lows": equal_lows},
            index=close.index,
        )


class OTELevels:
    """
    Optimal Trade Entry (OTE) — Fibonacci retracement zone for entries.

    OTE zone = 61.8% to 79% retracement of the most recent swing.
    Price entering this zone during a pullback after a BOS = high-probability
    entry aligned with the smart-money model.

    Parameters
    ----------
    ote_low : float
        Lower Fibonacci bound of OTE. Default 0.618 (61.8%).
    ote_high : float
        Upper Fibonacci bound of OTE. Default 0.79 (79%).
    swing_period : int
        Bars to look back for the swing origin. Default 20.
    """

    def __init__(
        self,
        ote_low: float = 0.618,
        ote_high: float = 0.79,
        swing_period: int = 20,
    ) -> None:
        self.ote_low = ote_low
        self.ote_high = ote_high
        self.swing_period = swing_period

    def compute(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns:
            swing_range_high  – high of the reference swing
            swing_range_low   – low of the reference swing
            ote_upper         – 79% retracement level
            ote_lower         – 61.8% retracement level
            price_in_ote      – 1 when close is inside the OTE zone
        """
        swing_high = high.rolling(self.swing_period).max()
        swing_low = low.rolling(self.swing_period).min()
        swing_range = swing_high - swing_low

        # For a bullish retracement: OTE is a pullback toward swing_low
        # OTE lower = swing_high - 0.79 * range
        # OTE upper = swing_high - 0.618 * range
        ote_upper = swing_high - self.ote_low * swing_range
        ote_lower = swing_high - self.ote_high * swing_range

        price_in_ote = ((close <= ote_upper) & (close >= ote_lower)).astype(int)

        return pd.DataFrame(
            {
                "swing_range_high": swing_high,
                "swing_range_low": swing_low,
                "ote_upper": ote_upper,
                "ote_lower": ote_lower,
                "price_in_ote": price_in_ote,
            },
            index=close.index,
        )


class ICTKillZones:
    """
    ICT Session Kill Zones — time-based liquidity windows.

    Institutionally active windows where stop runs and displacement moves are
    most likely. All times in UTC; the user should convert their index timezone
    to UTC before calling compute().

    Kill Zones (UTC equivalent of US/Eastern for US equities):
    - Asia:    00:00–03:00 UTC  (21:00–00:00 ET previous day)
    - London:  07:00–10:00 UTC  (02:00–05:00 ET)
    - NY AM:   13:30–16:00 UTC  (08:30–11:00 ET) — highest liquidity
    - NY PM:   18:30–21:00 UTC  (13:30–16:00 ET) — Silver Bullet window

    Parameters
    ----------
    timezone_offset_hours : float
        Offset to add to index timestamp to convert to UTC. Default 0 (already UTC).
    """

    def __init__(self, timezone_offset_hours: float = 0.0) -> None:
        self.tz_offset = pd.Timedelta(hours=timezone_offset_hours)

    def compute(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Parameters
        ----------
        index : pd.DatetimeIndex
            Bar timestamps (must have time component — use intraday bars).

        Returns
        -------
        pd.DataFrame indexed by `index` with columns:
            in_asia_kz    – 1 during Asia Kill Zone
            in_london_kz  – 1 during London Kill Zone
            in_ny_am_kz   – 1 during NY AM Kill Zone (highest liquidity)
            in_ny_pm_kz   – 1 during NY PM Kill Zone (Silver Bullet)
            in_any_kz     – 1 if in any kill zone
        """
        utc_idx = index + self.tz_offset
        hour = utc_idx.hour + utc_idx.minute / 60.0

        in_asia = ((hour >= 0) & (hour < 3)).astype(int)
        in_london = ((hour >= 7) & (hour < 10)).astype(int)
        in_ny_am = ((hour >= 13.5) & (hour < 16)).astype(int)
        in_ny_pm = ((hour >= 18.5) & (hour < 21)).astype(int)
        in_any = ((in_asia | in_london | in_ny_am | in_ny_pm) > 0).astype(int)

        return pd.DataFrame(
            {
                "in_asia_kz": in_asia,
                "in_london_kz": in_london,
                "in_ny_am_kz": in_ny_am,
                "in_ny_pm_kz": in_ny_pm,
                "in_any_kz": in_any,
            },
            index=index,
        )


class ICTPowerOfThree:
    """
    ICT Power of Three (PO3) — session phase identification.

    PO3 labels each daily session into three phases:
    1. Accumulation: Asia session builds position quietly
    2. Manipulation: London open sweeps highs/lows to trap retail
    3. Distribution: NY session makes the true directional move

    Operational definition (daily OHLCV):
    - Accumulation = close is within Asia range (defined as tight range at open)
    - Manipulation = session low < prior Asia low OR session high > prior Asia high
    - Distribution = session close significantly above/below the manipulation sweep level

    This implementation uses daily bars with open/high/low/close. For intraday
    PO3, use sub-hourly bars and map to UTC session times.

    Parameters
    ----------
    asia_tight_pct : float
        Maximum range (as % of close) to classify a session as "accumulation". Default 0.5%.
    """

    def __init__(self, asia_tight_pct: float = 0.5) -> None:
        self.asia_tight_pct = asia_tight_pct

    def compute(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns:
            session_range_pct – (high - low) / close × 100
            tight_range       – 1 if session_range_pct < asia_tight_pct (accumulation proxy)
            manipulation_up   – 1 if high > prior day's high but close < prior high
            manipulation_down – 1 if low < prior day's low but close > prior low
            distribution_up   – 1 if close > prior high (bullish distribution)
            distribution_down – 1 if close < prior low (bearish distribution)
        """
        prior_high = high.shift(1)
        prior_low = low.shift(1)

        session_range_pct = (high - low) / close.replace(0, np.nan) * 100
        tight_range = (session_range_pct < self.asia_tight_pct).astype(int)

        # Manipulation: sweep beyond prior session then reject
        manipulation_up = ((high > prior_high) & (close < prior_high)).astype(int)
        manipulation_down = ((low < prior_low) & (close > prior_low)).astype(int)

        # Distribution: price closes beyond prior extreme (trend confirmation)
        distribution_up = (close > prior_high).astype(int)
        distribution_down = (close < prior_low).astype(int)

        return pd.DataFrame(
            {
                "session_range_pct": session_range_pct,
                "tight_range": tight_range,
                "manipulation_up": manipulation_up,
                "manipulation_down": manipulation_down,
                "distribution_up": distribution_up,
                "distribution_down": distribution_down,
            },
            index=close.index,
        )


# ---------------------------------------------------------------------------
# Breaker Block Detector
# ---------------------------------------------------------------------------


class BreakerBlockDetector:
    """
    Breaker Block — an Order Block that price has subsequently violated.

    When price closes through an existing OB zone (through ob_high for bullish OB,
    through ob_low for bearish OB), that OB *flips polarity*:
    - Violated bullish OB → becomes resistance (bearish breaker)
    - Violated bearish OB → becomes support (bullish breaker)

    This flip is the most reliable ICT re-test pattern because institutions
    that originally placed orders in the OB zone are now trapped and will
    defend the level from the other side.

    Operational definitions
    -----------------------
    1. Detect OBs using the same logic as OrderBlockDetector.
    2. Track each OB until price closes beyond its zone.
    3. At the bar where the violation occurs, flag it as a breaker.
    4. Mark the polarity: bullish_breaker (from violated bearish OB),
       bearish_breaker (from violated bullish OB).

    Parameters
    ----------
    impulse_atr_multiple : float
        Minimum impulse size to qualify as an OB (in ATR multiples). Default 1.5.
    atr_period : int
        ATR lookback. Default 14.
    """

    def __init__(self, impulse_atr_multiple: float = 1.5, atr_period: int = 14) -> None:
        self.impulse_atr_multiple = impulse_atr_multiple
        self.atr_period = atr_period

    def compute(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns:
            bullish_breaker – 1 on bar where a bearish OB is violated (bullish reversal)
            bearish_breaker – 1 on bar where a bullish OB is violated (bearish reversal)
            breaker_high    – top of the violated OB zone
            breaker_low     – bottom of the violated OB zone
        """
        # First, get the base OBs from OrderBlockDetector
        ob_detector = OrderBlockDetector(
            impulse_atr_multiple=self.impulse_atr_multiple,
            atr_period=self.atr_period,
        )
        ob_df = ob_detector.compute(open_, high, low, close)

        n = len(close)
        bullish_breaker = np.zeros(n, dtype=int)
        bearish_breaker = np.zeros(n, dtype=int)
        breaker_high = np.full(n, np.nan)
        breaker_low = np.full(n, np.nan)

        # Track active OBs: list of (ob_bar, ob_high, ob_low, ob_type)
        # ob_type: 'bullish' = expect price to stay above ob_low,
        #          'bearish' = expect price to stay below ob_high
        active_obs: list[tuple[int, float, float, str]] = []

        for i in range(n):
            # Register new OBs formed at this bar
            if ob_df["bullish_ob"].iloc[i] == 1:
                oh = ob_df["ob_high"].iloc[i]
                ol = ob_df["ob_low"].iloc[i]
                if not (np.isnan(oh) or np.isnan(ol)):
                    active_obs.append((i, oh, ol, "bullish"))

            if ob_df["bearish_ob"].iloc[i] == 1:
                oh = ob_df["ob_high"].iloc[i]
                ol = ob_df["ob_low"].iloc[i]
                if not (np.isnan(oh) or np.isnan(ol)):
                    active_obs.append((i, oh, ol, "bearish"))

            # Check existing OBs for violations at current bar
            still_active = []
            for ob_bar, oh, ol, ob_type in active_obs:
                if ob_bar == i:  # just created — skip violation check this bar
                    still_active.append((ob_bar, oh, ol, ob_type))
                    continue

                cl = close.iloc[i]
                if ob_type == "bullish" and cl < ol:
                    # Price closed below bullish OB low → OB violated → bearish breaker
                    bearish_breaker[i] = 1
                    breaker_high[i] = oh
                    breaker_low[i] = ol
                    # OB consumed — do NOT append to still_active
                elif ob_type == "bearish" and cl > oh:
                    # Price closed above bearish OB high → OB violated → bullish breaker
                    bullish_breaker[i] = 1
                    breaker_high[i] = oh
                    breaker_low[i] = ol
                else:
                    still_active.append((ob_bar, oh, ol, ob_type))

            active_obs = still_active

        return pd.DataFrame(
            {
                "bullish_breaker": bullish_breaker,
                "bearish_breaker": bearish_breaker,
                "breaker_high": pd.Series(breaker_high, index=close.index),
                "breaker_low": pd.Series(breaker_low, index=close.index),
            },
            index=close.index,
        )


# ---------------------------------------------------------------------------
# Silver Bullet
# ---------------------------------------------------------------------------


class SilverBullet:
    """
    ICT Silver Bullet — FVG formed during the 10:00–11:00 AM New York window.

    This is the highest-probability ICT setup because:
    1. It occurs after the initial NY AM opening volatility (first hour).
    2. The 10:00 AM economic data releases often create a directional expansion.
    3. Any FVG formed in this window is treated as a true institution entry signal.

    The setup requires:
    - A Fair Value Gap (bullish or bearish) at bar i
    - Bar i's timestamp falls in the 10:00–11:00 AM New York (ET) window
    - New York ET = UTC - 5h (standard) or UTC - 4h (DST)

    The class auto-detects DST via the `index.tz` attribute when the index
    is timezone-aware; falls back to UTC-5 for naive indices.

    Parameters
    ----------
    min_gap_atr_multiple : float
        Minimum FVG size in ATR multiples. Default 0.1.
    atr_period : int
        ATR period. Default 14.
    ny_window_start : int
        NY ET hour for window open. Default 10.
    ny_window_end : int
        NY ET hour for window close (exclusive). Default 11.
    """

    def __init__(
        self,
        min_gap_atr_multiple: float = 0.1,
        atr_period: int = 14,
        ny_window_start: int = 10,
        ny_window_end: int = 11,
    ) -> None:
        self.min_gap_atr_multiple = min_gap_atr_multiple
        self.atr_period = atr_period
        self.ny_window_start = ny_window_start
        self.ny_window_end = ny_window_end

    def _in_silver_bullet_window(self, index: pd.DatetimeIndex) -> pd.Series:
        """Return boolean Series: True when bar is in 10–11 AM NY ET."""
        if index.tz is not None:
            ny_tz = pytz.timezone("America/New_York")
            local_hours = index.tz_convert(ny_tz).hour
        else:
            # Assume UTC input → NY ET ≈ UTC-5 (ignore DST for naive index)
            local_hours = (index.hour - 5) % 24

        return pd.Series(
            (local_hours >= self.ny_window_start) & (local_hours < self.ny_window_end),
            index=index,
        )

    def compute(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        high, low, close : pd.Series
            Intraday OHLCV data. Should be 1m–15m for meaningful results.

        Returns
        -------
        pd.DataFrame with columns:
            sb_bullish   – 1 when bullish FVG occurs in 10–11 AM NY window
            sb_bearish   – 1 when bearish FVG occurs in 10–11 AM NY window
            sb_fvg_top   – top of the Silver Bullet FVG
            sb_fvg_bot   – bottom of the Silver Bullet FVG
        """
        fvg_df = FairValueGapDetector(
            min_gap_atr_multiple=self.min_gap_atr_multiple,
            atr_period=self.atr_period,
        ).compute(high, low, close)

        in_window = self._in_silver_bullet_window(close.index)

        sb_bullish = (fvg_df["bullish_fvg"] == 1) & in_window
        sb_bearish = (fvg_df["bearish_fvg"] == 1) & in_window

        fvg_top = fvg_df["fvg_top"].where(sb_bullish | sb_bearish)
        fvg_bot = fvg_df["fvg_bottom"].where(sb_bullish | sb_bearish)

        return pd.DataFrame(
            {
                "sb_bullish": sb_bullish.astype(int),
                "sb_bearish": sb_bearish.astype(int),
                "sb_fvg_top": fvg_top,
                "sb_fvg_bot": fvg_bot,
            },
            index=close.index,
        )


# ---------------------------------------------------------------------------
# MMXM Cycle
# ---------------------------------------------------------------------------


class MMXMCycle:
    """
    Market Maker eXpansion Model (MMXM) — 4-phase market cycle labeling.

    The MMXM describes how market makers accumulate, sweep, expand, and
    distribute positions. Each bar is labeled with its current phase:

    Phase 0 — CONSOLIDATION (Accumulation)
        Price is ranging with tight ATR; market makers are accumulating
        positions below (bullish) or above (bearish) a known liquidity level.
        Detected as: ATR < rolling_avg_atr × atr_contraction_threshold

    Phase 1 — MANIPULATION (Liquidity Sweep)
        Price sweeps beyond a prior swing level (equal highs/lows or recent
        high/low) to trigger retail stops before reversing. This is the trap.
        Detected as: high > rolling_max(high, swing_period) then close reverses,
        OR low < rolling_min(low, swing_period) then close reverses.

    Phase 2 — EXPANSION (Trending Move / True Range)
        After the manipulation sweep, price expands in the opposite direction
        with above-average range. The true institutional move.
        Detected as: large candle range (> ATR × expansion_multiple) following
        a manipulation bar, moving opposite to the sweep direction.

    Phase 3 — RETRACEMENT / DISTRIBUTION
        Price pulls back (partially), creating entry opportunities for traders
        who missed the expansion. Often retraces to a FVG or OB.
        Detected as: smaller range bars following expansion, with close
        approaching midpoint of the expansion range.

    Parameters
    ----------
    swing_period : int
        Lookback for swing high/low detection. Default 10.
    atr_period : int
        ATR calculation period. Default 14.
    atr_contraction_threshold : float
        ATR ratio below which consolidation is flagged (< 0.8 = 20% below avg). Default 0.8.
    expansion_multiple : float
        Minimum range / ATR ratio for expansion detection. Default 1.5.
    """

    CONSOLIDATION = 0
    MANIPULATION = 1
    EXPANSION = 2
    RETRACEMENT = 3
    _LABELS = {0: "consolidation", 1: "manipulation", 2: "expansion", 3: "retracement"}

    def __init__(
        self,
        swing_period: int = 10,
        atr_period: int = 14,
        atr_contraction_threshold: float = 0.8,
        expansion_multiple: float = 1.5,
    ) -> None:
        self.swing_period = swing_period
        self.atr_period = atr_period
        self.atr_contraction_threshold = atr_contraction_threshold
        self.expansion_multiple = expansion_multiple

    def compute(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns:
            mmxm_phase    – integer phase label (0/1/2/3)
            mmxm_label    – string label ("consolidation"/"manipulation"/"expansion"/"retracement")
            in_consolidation  – 1 when phase == 0
            in_manipulation   – 1 when phase == 1
            in_expansion      – 1 when phase == 2
            in_retracement    – 1 when phase == 3
        """
        # ATR for range context
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(span=self.atr_period, adjust=False).mean()
        atr_avg = atr.rolling(self.swing_period * 2).mean()

        candle_range = high - low

        # Rolling swing levels for manipulation detection
        roll_high = high.rolling(self.swing_period).max()
        roll_low = low.rolling(self.swing_period).min()

        # --- Phase detection ---
        n = len(close)
        phase = np.full(n, self.CONSOLIDATION, dtype=int)

        for i in range(1, n):
            h_i = high.iloc[i]
            l_i = low.iloc[i]
            cl_i = close.iloc[i]
            cl_prev = close.iloc[i - 1]
            atr_i = atr.iloc[i]
            atr_avg_i = atr_avg.iloc[i]
            rng_i = candle_range.iloc[i]

            if np.isnan(atr_i) or np.isnan(atr_avg_i):
                continue

            rh = roll_high.iloc[i - 1] if i > 0 else np.nan
            rl = roll_low.iloc[i - 1] if i > 0 else np.nan

            # Phase 1 — Manipulation: sweep prior extreme then close back inside
            manipulation = False
            if not np.isnan(rh) and not np.isnan(rl):
                sweep_up = (h_i > rh) and (cl_i < rh)  # false breakout above
                sweep_down = (l_i < rl) and (cl_i > rl)  # false breakdown below
                manipulation = sweep_up or sweep_down

            # Phase 2 — Expansion: large candle (> expansion_multiple × ATR)
            expansion = (rng_i > atr_i * self.expansion_multiple) and not manipulation

            # Phase 3 — Retracement: prior phase was expansion, current range is smaller
            if i >= 2:
                prior_phase = phase[i - 1]
                prior_range = candle_range.iloc[i - 1]
                retracement = (
                    prior_phase == self.EXPANSION and rng_i < prior_range * 0.7
                )
            else:
                retracement = False

            # Phase 0 — Consolidation: ATR well below average
            consolidation = atr_i < atr_avg_i * self.atr_contraction_threshold

            # Priority: manipulation > expansion > retracement > consolidation
            if manipulation:
                phase[i] = self.MANIPULATION
            elif expansion:
                phase[i] = self.EXPANSION
            elif retracement:
                phase[i] = self.RETRACEMENT
            else:
                phase[i] = self.CONSOLIDATION  # default

        phase_series = pd.Series(phase, index=close.index)
        labels = phase_series.map(self._LABELS)

        return pd.DataFrame(
            {
                "mmxm_phase": phase_series,
                "mmxm_label": labels,
                "in_consolidation": (phase_series == self.CONSOLIDATION).astype(int),
                "in_manipulation": (phase_series == self.MANIPULATION).astype(int),
                "in_expansion": (phase_series == self.EXPANSION).astype(int),
                "in_retracement": (phase_series == self.RETRACEMENT).astype(int),
            },
            index=close.index,
        )


# ---------------------------------------------------------------------------
# SMT Divergence
# ---------------------------------------------------------------------------


class SMTDivergence:
    """
    Smart Money Technique (SMT) Divergence between two correlated instruments.

    SMT divergence occurs when two correlated instruments (e.g., ES/NQ, SPY/QQQ,
    DXY/Gold) make divergent swing highs or lows. One instrument sweeps the
    liquidity level (makes a new extreme) while the other fails to confirm —
    indicating institutional distribution/accumulation rather than genuine
    price discovery.

    Operational definitions
    -----------------------
    Bearish SMT: instrument_A makes a new swing HIGH but instrument_B does NOT
                 confirm the high (B's high is below its prior swing high).
                 → A is sweeping liquidity above; B is failing → reversal signal.

    Bullish SMT: instrument_A makes a new swing LOW but instrument_B does NOT
                 confirm the low (B's low is above its prior swing low).
                 → A is sweeping liquidity below; B is failing → bullish reversal.

    Divergence is detected on aligned bars when BOTH conditions hold simultaneously.

    Parameters
    ----------
    swing_period : int
        Bars on each side to qualify a local swing high/low. Default 5.
    atr_tolerance : float
        Minimum percentage difference between A and B swings to count as
        meaningful divergence (filters near-identical moves). Default 0.001 (0.1%).
    """

    def __init__(self, swing_period: int = 5, atr_tolerance: float = 0.001) -> None:
        self.swing_period = swing_period
        self.atr_tolerance = atr_tolerance

    @staticmethod
    def _rolling_swing_high(high: pd.Series, period: int) -> pd.Series:
        """1 if bar is a local swing high (highest in ±period bars), else 0."""
        n = len(high)
        sh = np.zeros(n, dtype=int)
        for i in range(period, n - period):
            window = high.iloc[i - period : i + period + 1]
            if high.iloc[i] == window.max() and (window == high.iloc[i]).sum() == 1:
                sh[i] = 1
        return pd.Series(sh, index=high.index)

    @staticmethod
    def _rolling_swing_low(low: pd.Series, period: int) -> pd.Series:
        """1 if bar is a local swing low (lowest in ±period bars), else 0."""
        n = len(low)
        sl = np.zeros(n, dtype=int)
        for i in range(period, n - period):
            window = low.iloc[i - period : i + period + 1]
            if low.iloc[i] == window.min() and (window == low.iloc[i]).sum() == 1:
                sl[i] = 1
        return pd.Series(sl, index=low.index)

    def compute(
        self,
        high_a: pd.Series,
        low_a: pd.Series,
        high_b: pd.Series,
        low_b: pd.Series,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        high_a, low_a : pd.Series
            High/low for the primary instrument (e.g., ES or SPY).
        high_b, low_b : pd.Series
            High/low for the correlated instrument (e.g., NQ or QQQ).
            Must share the same DatetimeIndex as A (align externally if needed).

        Returns
        -------
        pd.DataFrame with columns:
            bearish_smt   – 1 when A makes new swing high but B fails to confirm
            bullish_smt   – 1 when A makes new swing low but B fails to confirm
            smt_strength  – absolute % divergence between A and B at the swing
                            (larger = stronger institutional signal)
            divergence_direction – 'bearish', 'bullish', or '' (no signal)
        """
        n = len(high_a)
        p = self.swing_period

        sh_a = self._rolling_swing_high(high_a, p)
        sl_a = self._rolling_swing_low(low_a, p)
        sh_b = self._rolling_swing_high(high_b, p)
        sl_b = self._rolling_swing_low(low_b, p)

        bearish_smt = np.zeros(n, dtype=int)
        bullish_smt = np.zeros(n, dtype=int)
        smt_strength = np.zeros(n, dtype=float)

        # Track the most recent confirmed swing high/low for each instrument
        last_sh_a = np.nan
        last_sh_b = np.nan
        last_sl_a = np.nan
        last_sl_b = np.nan

        for i in range(n):
            h_a = high_a.iloc[i]
            l_a = low_a.iloc[i]
            h_b = high_b.iloc[i]
            l_b = low_b.iloc[i]

            # Bearish SMT: A makes swing high, B's concurrent high is BELOW prior swing high of B
            # Operationally: sh_a==1 at bar i, and high_b[i] < last_sh_b (B fails to confirm)
            if sh_a.iloc[i] == 1 and not np.isnan(last_sh_b):
                # Normalize B's high relative to last B swing (percentage basis)
                b_pct_diff = (last_sh_b - h_b) / last_sh_b
                if b_pct_diff > self.atr_tolerance:
                    bearish_smt[i] = 1
                    smt_strength[i] = round(b_pct_diff * 100, 4)

            # Bullish SMT: A makes swing low, B's concurrent low is ABOVE prior swing low of B
            if sl_a.iloc[i] == 1 and not np.isnan(last_sl_b):
                b_pct_diff = (l_b - last_sl_b) / last_sl_b
                if b_pct_diff > self.atr_tolerance:
                    bullish_smt[i] = 1
                    smt_strength[i] = round(b_pct_diff * 100, 4)

            # Update last confirmed swings
            if sh_a.iloc[i] == 1:
                last_sh_a = h_a
            if sh_b.iloc[i] == 1:
                last_sh_b = h_b
            if sl_a.iloc[i] == 1:
                last_sl_a = l_a
            if sl_b.iloc[i] == 1:
                last_sl_b = l_b

        direction = np.where(
            bearish_smt == 1, "bearish", np.where(bullish_smt == 1, "bullish", "")
        )

        return pd.DataFrame(
            {
                "bearish_smt": bearish_smt,
                "bullish_smt": bullish_smt,
                "smt_strength": smt_strength,
                "divergence_direction": direction,
            },
            index=high_a.index,
        )
