"""
Implied Volatility Surface Construction and Interpolation.

Provides:
- IV surface from options chain data
- Bilinear interpolation on log-moneyness and sqrt(time)
- ATM IV, skew, and term structure extraction
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import interpolate
from loguru import logger


@dataclass
class IVPoint:
    """Single IV observation."""

    strike: float
    expiry_days: int
    iv: float
    delta: Optional[float] = None
    option_type: str = "call"  # 'call' or 'put'


@dataclass
class IVSurfaceMetrics:
    """IV surface summary metrics."""

    atm_iv_30d: float
    atm_iv_60d: float
    atm_iv_90d: float
    skew_25d_30d: float  # 25-delta put IV minus ATM IV
    skew_25d_60d: float
    term_structure_slope: float  # IV90 - IV30
    vol_of_vol: float  # Std of ATM IV across expiries


class IVSurface:
    """
    Implied Volatility Surface with bilinear interpolation.

    The surface is constructed on:
    - X axis: Log-moneyness = log(K/S)
    - Y axis: Sqrt(T) where T is time to expiry in years

    This parameterization is more stable for interpolation as:
    - Log-moneyness normalizes across different underlying prices
    - Sqrt(T) respects vol term structure better than linear time
    """

    def __init__(
        self,
        spot_price: float,
        risk_free_rate: float = 0.05,
    ):
        """
        Initialize IV surface.

        Args:
            spot_price: Current underlying price
            risk_free_rate: Risk-free rate for forward calculation
        """
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate

        # Surface data
        self._iv_points: List[IVPoint] = []
        self._interpolator: Optional[interpolate.RectBivariateSpline] = None
        self._call_interpolator: Optional[interpolate.RectBivariateSpline] = None
        self._put_interpolator: Optional[interpolate.RectBivariateSpline] = None

        # Cached grids
        self._log_moneyness_grid: Optional[np.ndarray] = None
        self._sqrt_time_grid: Optional[np.ndarray] = None
        self._iv_grid: Optional[np.ndarray] = None

    def build_from_chain(self, options_chain: pd.DataFrame) -> None:
        """
        Build IV surface from options chain data.

        Args:
            options_chain: DataFrame with columns:
                - strike: Strike price
                - expiry or dte: Expiration date or days to expiry
                - impliedVolatility or iv: Implied volatility
                - type or option_type: 'call' or 'put' (optional)
        """
        self._iv_points = []

        # Normalize column names
        df = options_chain.copy()

        # Handle expiry/dte
        if "dte" not in df.columns:
            if "expiry" in df.columns:
                df["dte"] = (pd.to_datetime(df["expiry"]) - pd.Timestamp.now()).dt.days
            elif "expiration" in df.columns:
                df["dte"] = (
                    pd.to_datetime(df["expiration"]) - pd.Timestamp.now()
                ).dt.days
            else:
                raise ValueError("Options chain must have 'dte' or 'expiry' column")

        # Handle IV column
        iv_col = None
        for col in ["impliedVolatility", "iv", "implied_volatility"]:
            if col in df.columns:
                iv_col = col
                break

        if iv_col is None:
            raise ValueError("Options chain must have implied volatility column")

        # Handle option type
        type_col = None
        for col in ["type", "option_type", "optionType"]:
            if col in df.columns:
                type_col = col
                break

        # Extract IV points
        for _, row in df.iterrows():
            strike = float(row["strike"])
            dte = int(row["dte"])
            iv = float(row[iv_col])

            # Skip invalid data
            if iv <= 0 or iv > 5 or dte <= 0:
                continue

            opt_type = str(row[type_col]).lower() if type_col else "call"
            delta = (
                float(row["delta"])
                if "delta" in row and pd.notna(row["delta"])
                else None
            )

            self._iv_points.append(
                IVPoint(
                    strike=strike,
                    expiry_days=dte,
                    iv=iv,
                    delta=delta,
                    option_type=opt_type,
                )
            )

        if len(self._iv_points) > 9:  # Need at least 3x3 grid
            self._build_interpolator()

    def _build_interpolator(self) -> None:
        """Build the bilinear interpolation grid."""
        if len(self._iv_points) < 9:
            logger.warning("Insufficient IV points for interpolation")
            return

        # Convert to arrays
        log_moneyness = np.array(
            [np.log(p.strike / self.spot_price) for p in self._iv_points]
        )
        sqrt_time = np.array([np.sqrt(p.expiry_days / 365) for p in self._iv_points])
        ivs = np.array([p.iv for p in self._iv_points])

        # Create unique sorted grids
        unique_lm = np.unique(log_moneyness)
        unique_st = np.unique(sqrt_time)

        # If we don't have a nice grid, use scattered interpolation
        if len(unique_lm) < 3 or len(unique_st) < 3:
            # Use linear interpolation for scattered data
            from scipy.interpolate import LinearNDInterpolator

            points = np.column_stack([log_moneyness, sqrt_time])
            self._interpolator = LinearNDInterpolator(points, ivs, fill_value=np.nan)
            self._log_moneyness_grid = unique_lm
            self._sqrt_time_grid = unique_st
            return

        # Try to build rectangular grid
        try:
            # Create grid
            lm_grid = np.linspace(log_moneyness.min(), log_moneyness.max(), 20)
            st_grid = np.linspace(sqrt_time.min(), sqrt_time.max(), 10)

            # Fit surface using thin plate spline for scattered data
            from scipy.interpolate import Rbf

            rbf = Rbf(log_moneyness, sqrt_time, ivs, function="thin_plate", smooth=0.1)

            # Create interpolation on regular grid
            LM, ST = np.meshgrid(lm_grid, st_grid)
            IV_grid = rbf(LM.ravel(), ST.ravel()).reshape(LM.shape)

            # Clip to reasonable range
            IV_grid = np.clip(IV_grid, 0.01, 3.0)

            # Create bivariate spline for fast lookup
            self._interpolator = interpolate.RectBivariateSpline(
                st_grid, lm_grid, IV_grid, kx=3, ky=3
            )

            self._log_moneyness_grid = lm_grid
            self._sqrt_time_grid = st_grid
            self._iv_grid = IV_grid

        except Exception as e:
            logger.warning(f"Failed to build smooth surface: {e}")
            # Fallback to linear
            from scipy.interpolate import LinearNDInterpolator

            points = np.column_stack([log_moneyness, sqrt_time])
            self._interpolator = LinearNDInterpolator(points, ivs, fill_value=np.nan)

    def interpolate(self, strike: float, dte: int) -> Optional[float]:
        """
        Interpolate IV for given strike and days to expiry.

        Args:
            strike: Strike price
            dte: Days to expiry

        Returns:
            Interpolated IV, or None if out of range
        """
        if self._interpolator is None:
            return self._get_nearest_iv(strike, dte)

        log_moneyness = np.log(strike / self.spot_price)
        sqrt_time = np.sqrt(dte / 365)

        try:
            if hasattr(self._interpolator, "ev"):
                # RectBivariateSpline
                iv = float(self._interpolator.ev(sqrt_time, log_moneyness))
            else:
                # LinearNDInterpolator
                iv = float(self._interpolator(log_moneyness, sqrt_time))

            if np.isnan(iv) or iv <= 0:
                return self._get_nearest_iv(strike, dte)

            return np.clip(iv, 0.01, 3.0)

        except Exception:
            return self._get_nearest_iv(strike, dte)

    def _get_nearest_iv(self, strike: float, dte: int) -> Optional[float]:
        """Get IV from nearest point as fallback."""
        if not self._iv_points:
            return None

        # Find closest point by weighted distance
        best_iv = None
        best_dist = float("inf")

        for point in self._iv_points:
            # Weighted distance
            strike_dist = abs(np.log(strike) - np.log(point.strike))
            time_dist = abs(dte - point.expiry_days) / 30  # Normalize to months
            dist = strike_dist + time_dist

            if dist < best_dist:
                best_dist = dist
                best_iv = point.iv

        return best_iv

    def get_atm_iv(self, dte: int) -> Optional[float]:
        """
        Get ATM implied volatility for given DTE.

        Args:
            dte: Days to expiry

        Returns:
            ATM IV or None if unavailable
        """
        return self.interpolate(self.spot_price, dte)

    def get_skew(self, dte: int, delta_target: float = 0.25) -> Optional[float]:
        """
        Get volatility skew for given DTE.

        Skew = 25-delta put IV - ATM IV

        Args:
            dte: Days to expiry
            delta_target: Target delta for OTM put (default 0.25)

        Returns:
            Skew (positive means puts more expensive), or None
        """
        atm_iv = self.get_atm_iv(dte)
        if atm_iv is None:
            return None

        # Approximate OTM put strike for 25-delta
        # Using delta approximation: K ≈ S * exp(-0.5 * sigma^2 * T + sigma * sqrt(T) * N^-1(delta))
        from scipy.stats import norm

        T = dte / 365
        sigma = atm_iv

        # For 25-delta put
        d1_target = norm.ppf(delta_target)  # ~-0.67 for 25-delta
        strike_25d_put = self.spot_price * np.exp(
            -0.5 * sigma**2 * T - sigma * np.sqrt(T) * abs(d1_target)
        )

        # Get IV at this strike
        put_iv = self.interpolate(strike_25d_put, dte)

        if put_iv is None:
            return None

        return put_iv - atm_iv

    def get_term_structure(self, dtes: Optional[List[int]] = None) -> pd.Series:
        """
        Get ATM IV term structure.

        Args:
            dtes: List of days to expiry (default: 7, 14, 30, 60, 90, 180)

        Returns:
            Series of ATM IV indexed by DTE
        """
        if dtes is None:
            dtes = [7, 14, 30, 60, 90, 180]

        term_structure = {}
        for dte in dtes:
            iv = self.get_atm_iv(dte)
            if iv is not None:
                term_structure[dte] = iv

        return pd.Series(term_structure, name="atm_iv")

    def get_metrics(self) -> Optional[IVSurfaceMetrics]:
        """
        Calculate comprehensive IV surface metrics.

        Returns:
            IVSurfaceMetrics or None if insufficient data
        """
        atm_30 = self.get_atm_iv(30)
        atm_60 = self.get_atm_iv(60)
        atm_90 = self.get_atm_iv(90)

        if atm_30 is None:
            return None

        skew_30 = self.get_skew(30) or 0.0
        skew_60 = self.get_skew(60) or 0.0

        # Term structure slope
        term_slope = (atm_90 - atm_30) if atm_90 else 0.0

        # Vol of vol from term structure
        term = self.get_term_structure()
        vol_of_vol = term.std() if len(term) > 1 else 0.0

        return IVSurfaceMetrics(
            atm_iv_30d=atm_30 or 0.0,
            atm_iv_60d=atm_60 or atm_30 or 0.0,
            atm_iv_90d=atm_90 or atm_60 or atm_30 or 0.0,
            skew_25d_30d=skew_30,
            skew_25d_60d=skew_60,
            term_structure_slope=term_slope,
            vol_of_vol=vol_of_vol,
        )

    def update_spot(self, new_spot: float) -> None:
        """Update spot price (requires rebuilding interpolator)."""
        self.spot_price = new_spot
        if self._iv_points:
            self._build_interpolator()

    def __repr__(self) -> str:
        return (
            f"IVSurface(spot={self.spot_price:.2f}, "
            f"points={len(self._iv_points)}, "
            f"atm_30d={self.get_atm_iv(30) or 'N/A'})"
        )


def build_iv_surface_from_chain(
    options_chain: pd.DataFrame,
    spot_price: float,
    risk_free_rate: float = 0.05,
) -> IVSurface:
    """
    Convenience function to build IV surface from options chain.

    Args:
        options_chain: DataFrame with options data
        spot_price: Current underlying price
        risk_free_rate: Risk-free rate

    Returns:
        Constructed IVSurface
    """
    surface = IVSurface(spot_price, risk_free_rate)
    surface.build_from_chain(options_chain)
    return surface


def extract_iv_features(surface: IVSurface) -> Dict[str, float]:
    """
    Extract trading features from IV surface.

    Args:
        surface: IVSurface object

    Returns:
        Dict of features for ML/RL models
    """
    metrics = surface.get_metrics()

    if metrics is None:
        return {
            "iv_30d": np.nan,
            "iv_60d": np.nan,
            "iv_90d": np.nan,
            "iv_skew_30d": np.nan,
            "iv_skew_60d": np.nan,
            "iv_term_slope": np.nan,
            "iv_vol_of_vol": np.nan,
        }

    return {
        "iv_30d": metrics.atm_iv_30d,
        "iv_60d": metrics.atm_iv_60d,
        "iv_90d": metrics.atm_iv_90d,
        "iv_skew_30d": metrics.skew_25d_30d,
        "iv_skew_60d": metrics.skew_25d_60d,
        "iv_term_slope": metrics.term_structure_slope,
        "iv_vol_of_vol": metrics.vol_of_vol,
    }
