"""
FRED-sourced macro features.

Institutional macro signals derived from FRED economic data. Each class
operates on pre-fetched pd.Series (from FREDFetcher) and returns a DataFrame
of derived signals.

Includes:
- RealYieldFeatures: TIPS yield + inflation breakeven → growth/value rotation
- CreditSpreadFeatures: HY OAS level/momentum → equity drawdown leading indicator
- CopperGoldRatio: Economic cycle proxy
- DXYMomentum: Dollar strength impacts
- MOVEIndex: Bond vol precedes equity vol
- EquityBondCorrelation: Diversification regime detection
- VolOfVol: Second derivative of volatility — regime change precursor
"""

import numpy as np
import pandas as pd


class RealYieldFeatures:
    """
    Real yield and inflation breakeven signals.

    Rising real yields favor value; falling favor growth.
    Breakeven > 2.5% → inflation concern → favor commodities/value.

    Parameters
    ----------
    momentum_window : int
        Window for rate-of-change. Default 21.
    """

    def __init__(self, momentum_window: int = 21) -> None:
        self.momentum_window = momentum_window

    def compute(
        self, real_yield: pd.Series, breakeven: pd.Series
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        real_yield : pd.Series
            10Y TIPS yield (e.g. FRED DFII10).
        breakeven : pd.Series
            10Y breakeven inflation (e.g. FRED T10YIE).

        Returns
        -------
        pd.DataFrame with: real_yield_10y, breakeven_10y,
        real_yield_momentum, breakeven_momentum, growth_value_signal
        """
        ry_mom = real_yield.diff(self.momentum_window)
        be_mom = breakeven.diff(self.momentum_window)

        # Growth/value signal: high breakeven = favor value
        gv = pd.Series(0.0, index=real_yield.index)
        gv[breakeven > 2.5] = -1  # favor value
        gv[breakeven < 1.5] = 1  # favor growth

        return pd.DataFrame(
            {
                "real_yield_10y": real_yield,
                "breakeven_10y": breakeven,
                "real_yield_momentum": ry_mom,
                "breakeven_momentum": be_mom,
                "growth_value_signal": gv,
            },
            index=real_yield.index,
        )


class CreditSpreadFeatures:
    """
    High-yield credit spread signals.

    HY OAS widening is a leading indicator for equity drawdowns (2-4 weeks lead).

    Parameters
    ----------
    zscore_window : int
        Z-score lookback. Default 252.
    momentum_window : int
        Rate-of-change window. Default 21.
    """

    def __init__(self, zscore_window: int = 252, momentum_window: int = 21) -> None:
        self.zscore_window = zscore_window
        self.momentum_window = momentum_window

    def compute(self, hy_oas: pd.Series) -> pd.DataFrame:
        """
        Parameters
        ----------
        hy_oas : pd.Series
            HY OAS spread (e.g. FRED BAMLH0A0HYM2), in percentage points.

        Returns
        -------
        pd.DataFrame with: hy_oas, hy_oas_zscore, hy_oas_momentum, credit_regime
        """
        mean = hy_oas.rolling(self.zscore_window, min_periods=63).mean()
        std = hy_oas.rolling(self.zscore_window, min_periods=63).std()
        zscore = (hy_oas - mean) / std.replace(0, np.nan)
        momentum = hy_oas.diff(self.momentum_window)

        regime = pd.Series("NORMAL", index=hy_oas.index)
        regime[zscore < -1.0] = "TIGHT"
        regime[zscore > 1.0] = "WIDE"

        return pd.DataFrame(
            {
                "hy_oas": hy_oas,
                "hy_oas_zscore": zscore,
                "hy_oas_momentum": momentum,
                "credit_regime": regime,
            },
            index=hy_oas.index,
        )


class CopperGoldRatio:
    """
    Copper/gold ratio as economic cycle proxy.

    Rising ratio = expansion (copper demand up). Falling = contraction.

    Parameters
    ----------
    momentum_window : int
        Lookback for regime detection. Default 63.
    """

    def __init__(self, momentum_window: int = 63) -> None:
        self.momentum_window = momentum_window

    def compute(self, copper: pd.Series, gold: pd.Series) -> pd.DataFrame:
        ratio = copper / gold.replace(0, np.nan)
        momentum = ratio.pct_change(self.momentum_window)

        regime = pd.Series("NEUTRAL", index=copper.index)
        regime[momentum > 0.02] = "EXPANSION"
        regime[momentum < -0.02] = "CONTRACTION"

        return pd.DataFrame(
            {
                "copper_gold_ratio": ratio,
                "cg_ratio_momentum": momentum,
                "cg_regime": regime,
            },
            index=copper.index,
        )


class DXYMomentum:
    """
    Dollar index momentum signals.

    Strong dollar hurts multinationals and commodity plays.

    Parameters
    ----------
    zscore_window : int
        Default 63.
    momentum_window : int
        Default 21.
    """

    def __init__(self, zscore_window: int = 63, momentum_window: int = 21) -> None:
        self.zscore_window = zscore_window
        self.momentum_window = momentum_window

    def compute(self, dxy: pd.Series) -> pd.DataFrame:
        mean = dxy.rolling(self.zscore_window, min_periods=21).mean()
        std = dxy.rolling(self.zscore_window, min_periods=21).std()
        zscore = (dxy - mean) / std.replace(0, np.nan)
        momentum = dxy.pct_change(self.momentum_window)

        regime = pd.Series("NEUTRAL", index=dxy.index)
        regime[zscore > 1.0] = "STRONG"
        regime[zscore < -1.0] = "WEAK"

        return pd.DataFrame(
            {
                "dxy_level": dxy,
                "dxy_zscore_63": zscore,
                "dxy_momentum_21d": momentum,
                "dxy_regime": regime,
            },
            index=dxy.index,
        )


class MOVEIndex:
    """
    MOVE index (bond volatility) signals.

    Rising MOVE precedes equity vol expansion by 1-2 weeks.

    Parameters
    ----------
    zscore_window : int
        Default 252.
    elevated_threshold : float
        MOVE > this = elevated bond vol. Default 100.
    """

    def __init__(self, zscore_window: int = 252, elevated_threshold: float = 100.0) -> None:
        self.zscore_window = zscore_window
        self.elevated_threshold = elevated_threshold

    def compute(self, move: pd.Series) -> pd.DataFrame:
        mean = move.rolling(self.zscore_window, min_periods=63).mean()
        std = move.rolling(self.zscore_window, min_periods=63).std()
        zscore = (move - mean) / std.replace(0, np.nan)
        momentum = move.pct_change(21)
        elevated = (move > self.elevated_threshold).astype(int)

        return pd.DataFrame(
            {
                "move_level": move,
                "move_zscore": zscore,
                "move_momentum": momentum,
                "move_elevated": elevated,
            },
            index=move.index,
        )


class EquityBondCorrelation:
    """
    Equity-bond correlation regime.

    Normal: negative correlation (stocks down → bonds up = diversification works).
    Crisis: positive correlation (both sell off = diversification breaks).

    Parameters
    ----------
    window : int
        Rolling correlation window. Default 60.
    """

    def __init__(self, window: int = 60) -> None:
        self.window = window

    def compute(
        self, equity_returns: pd.Series, bond_returns: pd.Series
    ) -> pd.DataFrame:
        corr = equity_returns.rolling(self.window, min_periods=21).corr(bond_returns)
        corr_mom = corr.diff(21)

        regime = pd.Series("NORMAL", index=equity_returns.index)
        regime[corr > 0.0] = "CRISIS"

        return pd.DataFrame(
            {
                "eq_bond_corr_60d": corr,
                "eq_bond_corr_momentum": corr_mom,
                "eq_bond_corr_regime": regime,
            },
            index=equity_returns.index,
        )


class VolOfVol:
    """
    Volatility of volatility.

    Second derivative of volatility. Spikes precede regime changes.

    Parameters
    ----------
    vol_window : int
        Window for base volatility. Default 21.
    vov_window : int
        Window for vol-of-vol. Default 21.
    zscore_window : int
        Z-score lookback. Default 252.
    """

    def __init__(
        self, vol_window: int = 21, vov_window: int = 21, zscore_window: int = 252
    ) -> None:
        self.vol_window = vol_window
        self.vov_window = vov_window
        self.zscore_window = zscore_window

    def compute(self, close: pd.Series) -> pd.DataFrame:
        """
        Parameters
        ----------
        close : pd.Series
            Equity close prices (or VIX close for direct vol-of-vol).

        Returns
        -------
        pd.DataFrame with: vov, vov_zscore, vov_spike
        """
        returns = close.pct_change()
        vol = returns.rolling(self.vol_window, min_periods=5).std()
        vov = vol.rolling(self.vov_window, min_periods=5).std()

        mean = vov.rolling(self.zscore_window, min_periods=63).mean()
        std = vov.rolling(self.zscore_window, min_periods=63).std()
        zscore = (vov - mean) / std.replace(0, np.nan)

        spike = (zscore > 2.0).astype(int)

        return pd.DataFrame(
            {"vov": vov, "vov_zscore": zscore, "vov_spike": spike},
            index=close.index,
        )
