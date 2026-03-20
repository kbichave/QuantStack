"""
Institutional-grade momentum factors.

Academic momentum factor construction used by AQR, Two Sigma, and similar
quant funds. Distinct from DualMomentum (which is single-asset absolute
momentum per Antonacci).

Includes:
- InstitutionalMomentumFactors: Cross-sectional 12-1, residual, vol-adjusted,
  intermediate 7-1 momentum (Jegadeesh-Titman, Novy-Marx, Daniel-Moskowitz, Blitz)
- CrossSectionalDispersion: Universe-level dispersion and avg correlation
"""

import numpy as np
import pandas as pd


class InstitutionalMomentumFactors:
    """
    Academic momentum factors for single-symbol and cross-sectional use.

    Factors:
    - mom_7_1: Intermediate-term momentum (Novy-Marx 2012) — 7-month return skipping 1 month
    - residual_momentum: 12-1 return minus beta × market return (Blitz et al. 2011)
    - vol_adjusted_momentum: mom_12_1 / realized_vol (Daniel-Moskowitz 2016)

    Parameters
    ----------
    lookback : int
        Long momentum lookback in trading days. Default 252 (~12 months).
    skip : int
        Skip-month in trading days. Default 21 (~1 month).
    vol_lookback : int
        Lookback for realized vol in vol-adjusted momentum. Default 252.
    intermediate_lookback : int
        Intermediate-term lookback in trading days. Default 147 (~7 months).
    """

    def __init__(
        self,
        lookback: int = 252,
        skip: int = 21,
        vol_lookback: int = 252,
        intermediate_lookback: int = 147,
    ) -> None:
        self.lookback = lookback
        self.skip = skip
        self.vol_lookback = vol_lookback
        self.intermediate_lookback = intermediate_lookback

    def compute_single(
        self, close: pd.Series, market_close: pd.Series | None = None
    ) -> pd.DataFrame:
        """
        Single-symbol momentum factors.

        Parameters
        ----------
        close : pd.Series
            Asset close prices.
        market_close : pd.Series, optional
            Market benchmark (e.g. SPY) for residual momentum. If None,
            residual_momentum is NaN.

        Returns
        -------
        pd.DataFrame with columns:
            mom_12_1             – 12-month-minus-1-month return
            mom_7_1              – 7-month-minus-1-month return (Novy-Marx)
            residual_momentum    – Idiosyncratic momentum (beta-adjusted)
            vol_adjusted_momentum – mom_12_1 / realized_vol_12m
        """
        returns = close.pct_change()

        # 12-1 momentum
        ret_12m = close / close.shift(self.lookback) - 1
        ret_1m = close / close.shift(self.skip) - 1
        mom_12_1 = ret_12m - ret_1m

        # 7-1 intermediate momentum (Novy-Marx 2012)
        ret_7m = close / close.shift(self.intermediate_lookback) - 1
        mom_7_1 = ret_7m - ret_1m

        # Realized vol for vol-adjusted
        rvol = returns.rolling(self.vol_lookback, min_periods=self.vol_lookback // 2).std() * np.sqrt(252)
        vol_adj = mom_12_1 / rvol.replace(0, np.nan)

        # Residual momentum (requires market benchmark)
        residual = pd.Series(np.nan, index=close.index)
        if market_close is not None:
            mkt_ret = market_close.pct_change()
            # Align indices
            common = returns.index.intersection(mkt_ret.index)
            if len(common) > 63:
                r_aligned = returns.reindex(common)
                m_aligned = mkt_ret.reindex(common)
                # Rolling beta
                cov = r_aligned.rolling(self.lookback, min_periods=63).cov(m_aligned)
                mkt_var = m_aligned.rolling(self.lookback, min_periods=63).var()
                beta = cov / mkt_var.replace(0, np.nan)
                # Residual = asset 12-1 return - beta × market 12-1 return
                mkt_12_1 = market_close.reindex(common) / market_close.reindex(common).shift(self.lookback) - 1
                mkt_1m = market_close.reindex(common) / market_close.reindex(common).shift(self.skip) - 1
                mkt_mom = mkt_12_1 - mkt_1m
                residual = (mom_12_1.reindex(common) - beta * mkt_mom).reindex(close.index)

        return pd.DataFrame(
            {
                "mom_12_1": mom_12_1,
                "mom_7_1": mom_7_1,
                "residual_momentum": residual,
                "vol_adjusted_momentum": vol_adj,
            },
            index=close.index,
        )

    def compute_cross_section(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-sectional momentum ranking.

        Parameters
        ----------
        returns_df : pd.DataFrame
            DataFrame of close prices, columns = symbols.

        Returns
        -------
        pd.DataFrame with columns per symbol:
            {symbol}_cs_rank – Percentile rank of 12-1 momentum within universe
        """
        mom = returns_df / returns_df.shift(self.lookback) - 1
        skip_ret = returns_df / returns_df.shift(self.skip) - 1
        mom_12_1 = mom - skip_ret
        ranks = mom_12_1.rank(axis=1, pct=True)
        ranks.columns = [f"{c}_cs_rank" for c in ranks.columns]
        return ranks


class CrossSectionalDispersion:
    """
    Cross-sectional dispersion and average correlation of a symbol universe.

    High dispersion = stock-picking environment.
    High avg correlation = systematic risk dominates.

    Parameters
    ----------
    window : int
        Rolling window for computation. Default 21.
    min_symbols : int
        Minimum symbols required for meaningful output. Default 5.
    """

    def __init__(self, window: int = 21, min_symbols: int = 5) -> None:
        self.window = window
        self.min_symbols = min_symbols

    def compute(self, closes: dict[str, pd.Series]) -> pd.DataFrame:
        """
        Parameters
        ----------
        closes : dict[str, pd.Series]
            Symbol → close price series.

        Returns
        -------
        pd.DataFrame with columns:
            cs_dispersion        – Rolling std of cross-sectional returns
            cs_dispersion_zscore – Z-score vs 252-day mean
            cs_correlation_mean  – Average pairwise correlation
        """
        if len(closes) < self.min_symbols:
            # Not enough symbols for meaningful cross-sectional stats
            idx = list(closes.values())[0].index if closes else pd.DatetimeIndex([])
            return pd.DataFrame(
                {
                    "cs_dispersion": np.nan,
                    "cs_dispersion_zscore": np.nan,
                    "cs_correlation_mean": np.nan,
                },
                index=idx,
            )

        # Build returns DataFrame
        ret_df = pd.DataFrame({sym: s.pct_change() for sym, s in closes.items()})

        # Cross-sectional dispersion: std across symbols per bar
        cs_std = ret_df.std(axis=1)
        dispersion = cs_std.rolling(self.window, min_periods=self.window // 2).mean()

        disp_mean = dispersion.rolling(252, min_periods=63).mean()
        disp_std_roll = dispersion.rolling(252, min_periods=63).std()
        disp_z = (dispersion - disp_mean) / disp_std_roll.replace(0, np.nan)

        # Average pairwise correlation
        corr_mean = pd.Series(np.nan, index=ret_df.index)
        for i in range(self.window, len(ret_df)):
            window_data = ret_df.iloc[i - self.window + 1 : i + 1].dropna(axis=1, how="all")
            if window_data.shape[1] >= self.min_symbols:
                corr_matrix = window_data.corr()
                # Upper triangle mean (excluding diagonal)
                mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
                corr_mean.iloc[i] = corr_matrix.values[mask].mean()

        return pd.DataFrame(
            {
                "cs_dispersion": dispersion,
                "cs_dispersion_zscore": disp_z,
                "cs_correlation_mean": corr_mean,
            },
            index=ret_df.index,
        )
