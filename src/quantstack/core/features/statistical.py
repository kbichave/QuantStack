"""
Statistical features for regime detection and persistence analysis.

Institutional-grade features used by quant hedge funds (RenTech, AQR, Man AHL)
for identifying trending vs mean-reverting regimes, estimating mean-reversion
speed, and measuring time-series predictability.

Includes:
- YangZhangVolatility: Superior OHLC vol estimator (Yang & Zhang 2000)
- HurstExponent: Persistence measure via rescaled range (R/S) analysis
- VarianceRatioTest: Lo-MacKinlay (1988) random walk test
- OUHalfLife: Ornstein-Uhlenbeck mean-reversion speed estimation
- AutocorrelationSpectrum: Rolling ACF profile for momentum/reversal detection
- EntropyFeatures: Shannon + sample entropy for predictability assessment
"""

import math

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Yang-Zhang Volatility (+ Garman-Klass, Parkinson)
# ---------------------------------------------------------------------------


class YangZhangVolatility:
    """
    Yang-Zhang (2000) OHLC volatility estimator.

    Combines overnight (close-to-open), open-to-close, and Rogers-Satchell
    components. More efficient than close-to-close realized vol when OHLC
    data is available.

    Also provides Garman-Klass and Parkinson estimators for comparison.

    Parameters
    ----------
    period : int
        Rolling window for volatility estimation. Default 22 (~1 month).
    """

    def __init__(self, period: int = 22) -> None:
        self.period = period

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
            yang_zhang_vol   – Yang-Zhang annualized volatility
            garman_klass_vol – Garman-Klass annualized volatility
            parkinson_vol    – Parkinson annualized volatility
            yz_vs_close_ratio – YZ vol / close-to-close vol (>1 = YZ captures more)
        """
        n = self.period
        prev_close = close.shift(1)

        # Log returns
        log_oc = np.log(open_ / prev_close)  # overnight
        log_co = np.log(close / open_)  # open-to-close
        log_ho = np.log(high / open_)
        log_lo = np.log(low / open_)
        log_hc = np.log(high / close)
        log_lc = np.log(low / close)
        log_hl = np.log(high / low)
        log_cc = np.log(close / prev_close)

        # --- Rogers-Satchell component ---
        rs = log_ho * log_hc + log_lo * log_lc
        sigma2_rs = rs.rolling(n, min_periods=max(n // 2, 2)).mean()

        # --- Overnight variance ---
        sigma2_overnight = (log_oc**2).rolling(n, min_periods=max(n // 2, 2)).mean()

        # --- Open-to-close variance ---
        sigma2_close = (log_co**2).rolling(n, min_periods=max(n // 2, 2)).mean()

        # --- Yang-Zhang combination ---
        k = 0.34 / (1.34 + (n + 1) / max(n - 1, 1))
        sigma2_yz = sigma2_overnight + k * sigma2_close + (1 - k) * sigma2_rs
        yang_zhang = np.sqrt(sigma2_yz.clip(lower=0) * 252)

        # --- Garman-Klass ---
        gk_daily = 0.5 * log_hl**2 - (2 * math.log(2) - 1) * log_co**2
        sigma2_gk = gk_daily.rolling(n, min_periods=max(n // 2, 2)).mean()
        garman_klass = np.sqrt(sigma2_gk.clip(lower=0) * 252)

        # --- Parkinson ---
        park_daily = log_hl**2 / (4 * math.log(2))
        sigma2_park = park_daily.rolling(n, min_periods=max(n // 2, 2)).mean()
        parkinson = np.sqrt(sigma2_park.clip(lower=0) * 252)

        # --- Close-to-close realized vol ---
        sigma_cc = log_cc.rolling(n, min_periods=max(n // 2, 2)).std() * np.sqrt(252)

        ratio = np.where(sigma_cc > 0, yang_zhang / sigma_cc, np.nan)

        return pd.DataFrame(
            {
                "yang_zhang_vol": yang_zhang,
                "garman_klass_vol": garman_klass,
                "parkinson_vol": parkinson,
                "yz_vs_close_ratio": ratio,
            },
            index=close.index,
        )


# ---------------------------------------------------------------------------
# Hurst Exponent
# ---------------------------------------------------------------------------


class HurstExponent:
    """
    Hurst exponent via rescaled range (R/S) analysis.

    H > 0.5 → persistent (trending), H < 0.5 → anti-persistent (mean-reverting),
    H ≈ 0.5 → random walk.

    Parameters
    ----------
    window : int
        Rolling window for R/S computation. Default 252.
    min_lags : int
        Smallest sub-period length. Default 10.
    max_lags : int
        Largest sub-period length. Default 100.
    """

    def __init__(
        self, window: int = 252, min_lags: int = 10, max_lags: int = 100
    ) -> None:
        self.window = window
        self.min_lags = min_lags
        self.max_lags = max_lags

    @staticmethod
    def _rs_for_lag(series: np.ndarray, lag: int) -> float:
        """Compute average R/S statistic for a given lag (sub-period) length."""
        n = len(series)
        n_blocks = n // lag
        if n_blocks < 1:
            return np.nan
        rs_values = []
        for i in range(n_blocks):
            block = series[i * lag : (i + 1) * lag]
            mean_block = np.mean(block)
            cumdev = np.cumsum(block - mean_block)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(block, ddof=1)
            if s > 1e-15:
                rs_values.append(r / s)
        if not rs_values:
            return np.nan
        return np.mean(rs_values)

    def _hurst_from_array(self, returns: np.ndarray) -> float:
        """Estimate Hurst exponent from a returns array."""
        lags = []
        rs_stats = []
        lag = self.min_lags
        while lag <= min(self.max_lags, len(returns) // 2):
            rs_val = self._rs_for_lag(returns, lag)
            if not np.isnan(rs_val) and rs_val > 0:
                lags.append(lag)
                rs_stats.append(rs_val)
            lag = int(lag * 1.5) if lag < 20 else lag + 10

        if len(lags) < 3:
            return np.nan

        log_lags = np.log(lags)
        log_rs = np.log(rs_stats)
        slope, _, _, _, _ = sp_stats.linregress(log_lags, log_rs)
        return float(np.clip(slope, 0.0, 1.0))

    def compute(self, close: pd.Series) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns:
            hurst_exponent – H ∈ [0, 1]
            hurst_regime   – -1 (mean-reverting, H<0.4), 0 (random, 0.4≤H≤0.6), 1 (trending, H>0.6)
        """
        returns = close.pct_change().values
        n = len(returns)
        hurst = np.full(n, np.nan)

        for i in range(self.window, n):
            window_returns = returns[i - self.window + 1 : i + 1]
            valid = window_returns[~np.isnan(window_returns)]
            if len(valid) >= self.min_lags * 2:
                hurst[i] = self._hurst_from_array(valid)

        hurst_series = pd.Series(hurst, index=close.index)
        regime = pd.Series(np.nan, index=close.index)
        regime[hurst_series < 0.4] = -1
        regime[(hurst_series >= 0.4) & (hurst_series <= 0.6)] = 0
        regime[hurst_series > 0.6] = 1

        return pd.DataFrame(
            {"hurst_exponent": hurst_series, "hurst_regime": regime},
            index=close.index,
        )


# ---------------------------------------------------------------------------
# Variance Ratio Test
# ---------------------------------------------------------------------------


class VarianceRatioTest:
    """
    Lo-MacKinlay (1988) variance ratio test.

    VR(q) = Var(r_q) / (q × Var(r_1)).
    Under random walk: VR = 1. VR > 1 → momentum. VR < 1 → mean-reversion.

    Parameters
    ----------
    lags : list[int]
        Lag periods to compute variance ratios. Default [2, 5, 10, 20].
    window : int
        Rolling window for variance estimation. Default 126.
    """

    def __init__(self, lags: list[int] | None = None, window: int = 126) -> None:
        self.lags = lags or [2, 5, 10, 20]
        self.window = window

    def compute(self, close: pd.Series) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns:
            vr_2, vr_5, vr_10, vr_20 – Variance ratio at each lag
            vr_zscore_5              – Z-score of VR(5) vs 1.0
        """
        log_prices = np.log(close)
        r1 = log_prices.diff()
        var1 = r1.rolling(self.window, min_periods=self.window // 2).var()

        result = {}
        for q in self.lags:
            rq = log_prices.diff(q)
            varq = rq.rolling(self.window, min_periods=self.window // 2).var()
            vr = varq / (q * var1)
            result[f"vr_{q}"] = vr

        # Z-score for VR(5): approximate std under null = sqrt(2*(q-1)/(3*q*T))
        if 5 in self.lags:
            q = 5
            t = self.window
            null_std = np.sqrt(2 * (q - 1) / (3 * q * t)) if t > 0 else 1.0
            result["vr_zscore_5"] = (result["vr_5"] - 1.0) / null_std

        return pd.DataFrame(result, index=close.index)


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck Half-Life
# ---------------------------------------------------------------------------


class OUHalfLife:
    """
    Ornstein-Uhlenbeck mean-reversion speed estimation.

    Fits dX = (θμ - θX) dt + σdW via OLS regression:
    ΔX_t = α + β × X_{t-1} + ε
    θ = -β, half_life = -ln(2) / β

    Parameters
    ----------
    window : int
        Rolling window for OLS regression. Default 126.
    """

    def __init__(self, window: int = 126) -> None:
        self.window = window

    def compute(self, series: pd.Series) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns:
            ou_half_life      – Mean-reversion half-life in bars (NaN if not mean-reverting)
            ou_theta          – Mean-reversion speed parameter
            ou_mu             – Equilibrium level
            ou_half_life_valid – 1 if β < 0 (mean-reverting), else 0
        """
        dx = series.diff()
        x_lag = series.shift(1)
        n = len(series)

        half_life = np.full(n, np.nan)
        theta = np.full(n, np.nan)
        mu = np.full(n, np.nan)
        valid = np.full(n, 0.0)

        for i in range(self.window, n):
            y = dx.iloc[i - self.window + 1 : i + 1].values
            x = x_lag.iloc[i - self.window + 1 : i + 1].values
            mask = ~(np.isnan(y) | np.isnan(x))
            y_clean, x_clean = y[mask], x[mask]
            if len(y_clean) < 10:
                continue

            x_with_const = np.column_stack([np.ones(len(x_clean)), x_clean])
            try:
                coefs, _, _, _ = np.linalg.lstsq(x_with_const, y_clean, rcond=None)
            except np.linalg.LinAlgError:
                continue

            alpha, beta = coefs[0], coefs[1]
            if beta < -1e-10:
                half_life[i] = -math.log(2) / beta
                theta[i] = -beta
                mu[i] = -alpha / beta
                valid[i] = 1.0

        return pd.DataFrame(
            {
                "ou_half_life": half_life,
                "ou_theta": theta,
                "ou_mu": mu,
                "ou_half_life_valid": valid,
            },
            index=series.index,
        )


# ---------------------------------------------------------------------------
# Autocorrelation Spectrum
# ---------------------------------------------------------------------------


class AutocorrelationSpectrum:
    """
    Rolling autocorrelation at multiple lags.

    Momentum regimes have positive ACF at short lags; mean-reversion regimes
    have negative ACF. The decay rate of |ACF| vs lag distinguishes regimes.

    Parameters
    ----------
    window : int
        Rolling window for ACF estimation. Default 63.
    max_lag : int
        Maximum lag to compute. Default 20.
    """

    def __init__(self, window: int = 63, max_lag: int = 20) -> None:
        self.window = window
        self.max_lag = max_lag

    def compute(self, close: pd.Series) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns:
            acf_lag1      – Autocorrelation at lag 1
            acf_lag5      – Autocorrelation at lag 5
            acf_lag10     – Autocorrelation at lag 10
            acf_sum_1_5   – Sum of ACF lags 1-5 (positive = momentum, negative = reversal)
            acf_decay_rate – Slope of log|ACF| vs lag (faster decay = less predictable)
        """
        returns = close.pct_change()
        n = len(returns)
        acf1 = np.full(n, np.nan)
        acf5 = np.full(n, np.nan)
        acf10 = np.full(n, np.nan)
        acf_sum = np.full(n, np.nan)
        decay = np.full(n, np.nan)

        for i in range(self.window + self.max_lag, n):
            r = returns.iloc[i - self.window + 1 : i + 1].values
            if np.isnan(r).sum() > self.window * 0.3:
                continue
            r_clean = r[~np.isnan(r)]
            if len(r_clean) < self.max_lag + 5:
                continue

            r_dm = r_clean - np.mean(r_clean)
            var = np.sum(r_dm**2)
            if var < 1e-20:
                continue

            acfs = []
            for lag in range(1, self.max_lag + 1):
                c = np.sum(r_dm[lag:] * r_dm[:-lag]) / var
                acfs.append(c)

            acf1[i] = acfs[0]
            if len(acfs) >= 5:
                acf5[i] = acfs[4]
                acf_sum[i] = sum(acfs[:5])
            if len(acfs) >= 10:
                acf10[i] = acfs[9]

            # Decay rate: slope of log|ACF| vs lag for positive ACFs
            abs_acfs = [abs(a) for a in acfs if abs(a) > 1e-10]
            if len(abs_acfs) >= 3:
                log_abs = np.log(abs_acfs[:10])
                lag_idx = np.arange(1, len(log_abs) + 1, dtype=float)
                slope, _, _, _, _ = sp_stats.linregress(lag_idx, log_abs)
                decay[i] = slope

        return pd.DataFrame(
            {
                "acf_lag1": acf1,
                "acf_lag5": acf5,
                "acf_lag10": acf10,
                "acf_sum_1_5": acf_sum,
                "acf_decay_rate": decay,
            },
            index=close.index,
        )


# ---------------------------------------------------------------------------
# Entropy Features
# ---------------------------------------------------------------------------


class EntropyFeatures:
    """
    Shannon entropy, approximate entropy, and sample entropy of returns.

    Low entropy → predictable (trending). High entropy → random.

    Parameters
    ----------
    window : int
        Rolling window. Default 63.
    n_bins : int
        Bins for Shannon entropy discretization. Default 10.
    m : int
        Template length for ApEn/SampEn. Default 2.
    r_mult : float
        Tolerance multiplier (r = r_mult × std). Default 0.2.
    """

    def __init__(
        self, window: int = 63, n_bins: int = 10, m: int = 2, r_mult: float = 0.2
    ) -> None:
        self.window = window
        self.n_bins = n_bins
        self.m = m
        self.r_mult = r_mult

    @staticmethod
    def _shannon(arr: np.ndarray, n_bins: int) -> float:
        """Shannon entropy of binned array."""
        counts, _ = np.histogram(arr, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    @staticmethod
    def _sample_entropy(arr: np.ndarray, m: int, r: float) -> float:
        """Sample entropy (Richman & Moorman 2000)."""
        n = len(arr)
        if n < m + 2 or r <= 0:
            return np.nan

        def _count_matches(template_len: int) -> int:
            count = 0
            for i in range(n - template_len):
                for j in range(i + 1, n - template_len):
                    if (
                        np.max(
                            np.abs(
                                arr[i : i + template_len] - arr[j : j + template_len]
                            )
                        )
                        < r
                    ):
                        count += 1
            return count

        a = _count_matches(m + 1)
        b = _count_matches(m)
        if b == 0:
            return np.nan
        return -np.log(a / b) if a > 0 else np.nan

    def compute(self, close: pd.Series) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns:
            shannon_entropy  – Shannon entropy of binned returns
            sample_entropy   – Sample entropy (complexity measure)
            entropy_regime   – -1 (low entropy / trending), 0 (normal), 1 (high / random)
        """
        returns = close.pct_change().values
        n = len(returns)
        shannon = np.full(n, np.nan)
        sampen = np.full(n, np.nan)

        for i in range(self.window, n):
            r = returns[i - self.window + 1 : i + 1]
            valid = r[~np.isnan(r)]
            if len(valid) < self.window // 2:
                continue

            shannon[i] = self._shannon(valid, self.n_bins)

            std = np.std(valid)
            if std > 1e-15:
                sampen[i] = self._sample_entropy(valid, self.m, self.r_mult * std)

        shannon_s = pd.Series(shannon, index=close.index)
        sampen_s = pd.Series(sampen, index=close.index)

        # Regime from Shannon: use rolling z-score
        sh_mean = shannon_s.rolling(252, min_periods=63).mean()
        sh_std = shannon_s.rolling(252, min_periods=63).std()
        sh_z = (shannon_s - sh_mean) / sh_std.replace(0, np.nan)

        regime = pd.Series(np.nan, index=close.index)
        regime[sh_z < -1.0] = -1  # low entropy = trending
        regime[(sh_z >= -1.0) & (sh_z <= 1.0)] = 0
        regime[sh_z > 1.0] = 1  # high entropy = random

        return pd.DataFrame(
            {
                "shannon_entropy": shannon_s,
                "sample_entropy": sampen_s,
                "entropy_regime": regime,
            },
            index=close.index,
        )
