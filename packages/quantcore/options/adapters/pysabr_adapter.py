# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
PySABR adapter for SABR volatility surface fitting and interpolation.

Provides:
- SABR model calibration from market quotes
- Volatility smile interpolation
- Surface parameter extraction for trading signals

The SABR model (Stochastic Alpha Beta Rho) captures:
- Volatility smile/skew patterns
- Strike-dependent implied volatility
- More realistic vol dynamics than Black-Scholes
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SABRParams:
    """SABR model parameters."""

    alpha: float  # ATM volatility level
    beta: float  # CEV exponent (0=normal, 1=lognormal)
    rho: float  # Correlation between spot and vol
    volvol: float  # Volatility of volatility (nu)
    forward: float
    time_to_expiry: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "rho": self.rho,
            "volvol": self.volvol,
            "forward": self.forward,
            "time_to_expiry": self.time_to_expiry,
        }


def fit_sabr_surface(
    quotes: pd.DataFrame,
    forward: float,
    time_to_expiry: float,
    beta: float = 1.0,
    initial_guess: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Fit SABR model to market IV quotes.

    Args:
        quotes: DataFrame with columns:
            - strike: Strike prices
            - iv: Market implied volatilities
            - (optional) volume, open_interest for weighting
        forward: Forward price of underlying
        time_to_expiry: Time to expiration in years
        beta: CEV exponent, typically 0, 0.5, or 1.0 (default 1.0 for lognormal)
        initial_guess: Optional initial parameter guess

    Returns:
        Dictionary with:
            - params: SABRParams object
            - fit_quality: R-squared and RMSE
            - fitted_vols: Model IVs at input strikes
            - residuals: Fit errors
    """
    if quotes.empty:
        raise ValueError("Empty quotes DataFrame provided")

    required_cols = ["strike", "iv"]
    for col in required_cols:
        if col not in quotes.columns:
            raise ValueError(f"Missing required column: {col}")

    strikes = quotes["strike"].values
    market_vols = quotes["iv"].values

    # Filter invalid data
    valid_mask = (strikes > 0) & (market_vols > 0) & (market_vols < 5.0)
    strikes = strikes[valid_mask]
    market_vols = market_vols[valid_mask]

    if len(strikes) < 3:
        raise ValueError("Need at least 3 valid strike/IV pairs to fit SABR")

    # Calculate ATM vol as initial guess for alpha
    atm_idx = np.argmin(np.abs(strikes - forward))
    atm_vol = market_vols[atm_idx]

    # Initial parameter guess (needed for both pysabr and scipy fallback)
    if initial_guess:
        alpha0 = initial_guess.get("alpha", atm_vol)
        rho0 = initial_guess.get("rho", -0.3)
        volvol0 = initial_guess.get("volvol", 0.5)
    else:
        alpha0 = atm_vol
        rho0 = -0.3  # Typical negative correlation
        volvol0 = 0.5

    try:
        from pysabr import Hagan2002LognormalSABR, hagan_2002_lognormal_sabr

        # Fit SABR using pysabr's calibration
        sabr = Hagan2002LognormalSABR(
            f=forward,
            shift=0,
            t=time_to_expiry,
            beta=beta,
        )

        # Calibrate to market vols
        alpha, rho, volvol = sabr.fit(strikes, market_vols)

        # Create params object
        params = SABRParams(
            alpha=alpha,
            beta=beta,
            rho=rho,
            volvol=volvol,
            forward=forward,
            time_to_expiry=time_to_expiry,
        )

        # Calculate fitted vols and residuals
        fitted_vols = np.array(
            [
                hagan_2002_lognormal_sabr(
                    k, forward, time_to_expiry, alpha, beta, rho, volvol
                )
                for k in strikes
            ]
        )

        residuals = market_vols - fitted_vols
        rmse = np.sqrt(np.mean(residuals**2))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((market_vols - np.mean(market_vols)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "params": params,
            "params_dict": params.to_dict(),
            "fit_quality": {
                "r_squared": float(r_squared),
                "rmse": float(rmse),
                "n_points": len(strikes),
            },
            "fitted_vols": fitted_vols.tolist(),
            "residuals": residuals.tolist(),
            "strikes": strikes.tolist(),
            "market_vols": market_vols.tolist(),
        }

    except ImportError:
        logger.warning("pysabr not available, using scipy-based fitting")
        return _fit_sabr_scipy(
            strikes, market_vols, forward, time_to_expiry, beta, alpha0, rho0, volvol0
        )
    except Exception as e:
        logger.error(f"SABR fitting failed: {e}")
        raise


def _fit_sabr_scipy(
    strikes: np.ndarray,
    market_vols: np.ndarray,
    forward: float,
    time_to_expiry: float,
    beta: float,
    alpha0: float,
    rho0: float,
    volvol0: float,
) -> Dict[str, Any]:
    """Fallback SABR fitting using scipy optimization."""
    from scipy.optimize import minimize

    def sabr_vol_hagan(k, f, t, alpha, beta, rho, nu):
        """Hagan et al. (2002) SABR approximation formula."""
        if k <= 0 or f <= 0 or t <= 0 or alpha <= 0:
            return np.nan

        # At-the-money case
        if abs(k - f) < 1e-10:
            term1 = alpha / (f ** (1 - beta))
            term2 = (
                1
                + (
                    (1 - beta) ** 2 * alpha**2 / (24 * f ** (2 - 2 * beta))
                    + rho * beta * nu * alpha / (4 * f ** (1 - beta))
                    + (2 - 3 * rho**2) * nu**2 / 24
                )
                * t
            )
            return term1 * term2

        # General case
        fk = f * k
        log_fk = np.log(f / k)
        fk_beta = (fk) ** ((1 - beta) / 2)

        z = nu / alpha * fk_beta * log_fk
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

        if abs(x_z) < 1e-10:
            x_z = 1.0

        term1 = alpha / (
            fk_beta
            * (
                1
                + (1 - beta) ** 2 / 24 * log_fk**2
                + (1 - beta) ** 4 / 1920 * log_fk**4
            )
        )
        term2 = z / x_z
        term3 = (
            1
            + (
                (1 - beta) ** 2 * alpha**2 / (24 * fk ** (1 - beta))
                + rho * beta * nu * alpha / (4 * fk_beta)
                + (2 - 3 * rho**2) * nu**2 / 24
            )
            * t
        )

        return term1 * term2 * term3

    def objective(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
            return 1e10

        model_vols = np.array(
            [
                sabr_vol_hagan(k, forward, time_to_expiry, alpha, beta, rho, nu)
                for k in strikes
            ]
        )

        if np.any(np.isnan(model_vols)):
            return 1e10

        return np.sum((market_vols - model_vols) ** 2)

    # Optimize
    result = minimize(
        objective,
        x0=[alpha0, rho0, volvol0],
        bounds=[(0.001, 5.0), (-0.999, 0.999), (0.001, 5.0)],
        method="L-BFGS-B",
    )

    alpha, rho, volvol = result.x

    params = SABRParams(
        alpha=alpha,
        beta=beta,
        rho=rho,
        volvol=volvol,
        forward=forward,
        time_to_expiry=time_to_expiry,
    )

    fitted_vols = np.array(
        [
            sabr_vol_hagan(k, forward, time_to_expiry, alpha, beta, rho, volvol)
            for k in strikes
        ]
    )

    residuals = market_vols - fitted_vols
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((market_vols - np.mean(market_vols)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "params": params,
        "params_dict": params.to_dict(),
        "fit_quality": {
            "r_squared": float(r_squared),
            "rmse": float(rmse),
            "n_points": len(strikes),
        },
        "fitted_vols": fitted_vols.tolist(),
        "residuals": residuals.tolist(),
        "strikes": strikes.tolist(),
        "market_vols": market_vols.tolist(),
    }


def interpolate_sabr_vol(
    sabr_params: Union[SABRParams, Dict[str, float]],
    strike: float,
    forward: Optional[float] = None,
    time_to_expiry: Optional[float] = None,
) -> float:
    """
    Interpolate implied volatility using fitted SABR parameters.

    Args:
        sabr_params: SABRParams object or dict with alpha, beta, rho, volvol
        strike: Strike to interpolate
        forward: Forward price (uses params if not provided)
        time_to_expiry: Time to expiry (uses params if not provided)

    Returns:
        Interpolated implied volatility
    """
    if isinstance(sabr_params, SABRParams):
        alpha = sabr_params.alpha
        beta = sabr_params.beta
        rho = sabr_params.rho
        volvol = sabr_params.volvol
        forward = forward or sabr_params.forward
        time_to_expiry = time_to_expiry or sabr_params.time_to_expiry
    else:
        alpha = sabr_params["alpha"]
        beta = sabr_params["beta"]
        rho = sabr_params["rho"]
        volvol = sabr_params["volvol"]
        forward = forward or sabr_params.get("forward")
        time_to_expiry = time_to_expiry or sabr_params.get("time_to_expiry")

    if forward is None or time_to_expiry is None:
        raise ValueError("forward and time_to_expiry must be provided")

    try:
        from pysabr import hagan_2002_lognormal_sabr

        return float(
            hagan_2002_lognormal_sabr(
                strike, forward, time_to_expiry, alpha, beta, rho, volvol
            )
        )

    except ImportError:
        # Use internal Hagan formula
        return _hagan_sabr_vol(
            strike, forward, time_to_expiry, alpha, beta, rho, volvol
        )


def _hagan_sabr_vol(k, f, t, alpha, beta, rho, nu):
    """Hagan et al. (2002) SABR approximation formula."""
    if k <= 0 or f <= 0 or t <= 0 or alpha <= 0:
        return 0.0

    if abs(k - f) < 1e-10:
        term1 = alpha / (f ** (1 - beta))
        term2 = (
            1
            + (
                (1 - beta) ** 2 * alpha**2 / (24 * f ** (2 - 2 * beta))
                + rho * beta * nu * alpha / (4 * f ** (1 - beta))
                + (2 - 3 * rho**2) * nu**2 / 24
            )
            * t
        )
        return term1 * term2

    fk = f * k
    log_fk = np.log(f / k)
    fk_beta = (fk) ** ((1 - beta) / 2)

    z = nu / alpha * fk_beta * log_fk
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

    if abs(x_z) < 1e-10:
        x_z = 1.0

    term1 = alpha / (
        fk_beta
        * (1 + (1 - beta) ** 2 / 24 * log_fk**2 + (1 - beta) ** 4 / 1920 * log_fk**4)
    )
    term2 = z / x_z
    term3 = (
        1
        + (
            (1 - beta) ** 2 * alpha**2 / (24 * fk ** (1 - beta))
            + rho * beta * nu * alpha / (4 * fk_beta)
            + (2 - 3 * rho**2) * nu**2 / 24
        )
        * t
    )

    return term1 * term2 * term3


def get_sabr_smile(
    sabr_params: Union[SABRParams, Dict[str, float]],
    strikes: Union[List[float], np.ndarray],
    forward: Optional[float] = None,
    time_to_expiry: Optional[float] = None,
) -> pd.DataFrame:
    """
    Generate volatility smile from SABR parameters.

    Args:
        sabr_params: SABRParams object or dict
        strikes: Array of strikes to evaluate
        forward: Forward price
        time_to_expiry: Time to expiry

    Returns:
        DataFrame with strike and iv columns
    """
    strikes = np.atleast_1d(strikes)

    vols = np.array(
        [interpolate_sabr_vol(sabr_params, k, forward, time_to_expiry) for k in strikes]
    )

    return pd.DataFrame(
        {
            "strike": strikes,
            "iv": vols,
        }
    )


def get_sabr_skew_metrics(
    sabr_params: Union[SABRParams, Dict[str, float]],
    forward: Optional[float] = None,
    time_to_expiry: Optional[float] = None,
) -> Dict[str, float]:
    """
    Extract key skew metrics from SABR surface.

    Args:
        sabr_params: SABR parameters
        forward: Forward price
        time_to_expiry: Time to expiry

    Returns:
        Dictionary with:
            - atm_vol: At-the-money volatility
            - skew_25d: 25-delta skew (put vol - call vol)
            - skew_10d: 10-delta skew
            - smile_curvature: Butterfly measure
            - risk_reversal_25d: 25-delta risk reversal
    """
    if isinstance(sabr_params, dict):
        forward = forward or sabr_params.get("forward")
        time_to_expiry = time_to_expiry or sabr_params.get("time_to_expiry")
    else:
        forward = forward or sabr_params.forward
        time_to_expiry = time_to_expiry or sabr_params.time_to_expiry

    # Generate strikes at various delta levels (approximate)
    atm = forward
    k_90 = forward * 0.90  # ~25-delta put
    k_95 = forward * 0.95  # ~40-delta put
    k_105 = forward * 1.05  # ~40-delta call
    k_110 = forward * 1.10  # ~25-delta call
    k_85 = forward * 0.85  # ~10-delta put
    k_115 = forward * 1.15  # ~10-delta call

    atm_vol = interpolate_sabr_vol(sabr_params, atm, forward, time_to_expiry)
    vol_90 = interpolate_sabr_vol(sabr_params, k_90, forward, time_to_expiry)
    vol_95 = interpolate_sabr_vol(sabr_params, k_95, forward, time_to_expiry)
    vol_105 = interpolate_sabr_vol(sabr_params, k_105, forward, time_to_expiry)
    vol_110 = interpolate_sabr_vol(sabr_params, k_110, forward, time_to_expiry)
    vol_85 = interpolate_sabr_vol(sabr_params, k_85, forward, time_to_expiry)
    vol_115 = interpolate_sabr_vol(sabr_params, k_115, forward, time_to_expiry)

    # Skew = OTM put vol - OTM call vol
    skew_25d = vol_90 - vol_110
    skew_10d = vol_85 - vol_115

    # Risk reversal = call vol - put vol
    risk_reversal_25d = vol_110 - vol_90

    # Butterfly (smile curvature) = (OTM put + OTM call)/2 - ATM
    butterfly_25d = (vol_90 + vol_110) / 2 - atm_vol

    return {
        "atm_vol": float(atm_vol),
        "skew_25d": float(skew_25d),
        "skew_10d": float(skew_10d),
        "risk_reversal_25d": float(risk_reversal_25d),
        "butterfly_25d": float(butterfly_25d),
        "vol_90_moneyness": float(vol_90),
        "vol_110_moneyness": float(vol_110),
    }


def fit_term_structure_sabr(
    chain_data: pd.DataFrame,
    spot: float,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
    beta: float = 1.0,
) -> Dict[str, Any]:
    """
    Fit SABR surfaces across multiple expiries.

    Args:
        chain_data: DataFrame with strike, iv, dte columns
        spot: Current spot price
        risk_free_rate: Risk-free rate
        dividend_yield: Dividend yield
        beta: SABR beta parameter

    Returns:
        Dictionary with SABR params per expiry and term structure metrics
    """
    if "dte" not in chain_data.columns:
        raise ValueError("chain_data must have 'dte' column")

    results = {}
    expiries = sorted(chain_data["dte"].unique())

    for dte in expiries:
        if dte <= 0:
            continue

        expiry_data = chain_data[chain_data["dte"] == dte]
        time_to_expiry = dte / 365.0

        # Calculate forward price
        forward = spot * np.exp((risk_free_rate - dividend_yield) * time_to_expiry)

        try:
            fit_result = fit_sabr_surface(
                quotes=expiry_data,
                forward=forward,
                time_to_expiry=time_to_expiry,
                beta=beta,
            )
            results[int(dte)] = fit_result
        except Exception as e:
            logger.warning(f"Failed to fit SABR for DTE={dte}: {e}")
            continue

    # Extract term structure
    if results:
        atm_vols = {
            dte: (
                res["params"].alpha
                if isinstance(res["params"], SABRParams)
                else res["params_dict"]["alpha"]
            )
            for dte, res in results.items()
        }

        return {
            "expiry_fits": results,
            "atm_term_structure": atm_vols,
            "expiries": list(results.keys()),
        }

    return {"expiry_fits": {}, "atm_term_structure": {}, "expiries": []}
