"""
Options pricing using Black-Scholes model.

Provides:
- Black-Scholes pricing for calls and puts
- Black-Scholes-Merton with continuous dividends
- Greeks computation (delta, gamma, theta, vega, rho)
- Implied volatility solver
"""

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from quantcore.options.models import OptionType


@dataclass
class Greeks:
    """Container for option Greeks."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


# Default dividend yields for common tickers (annual continuous yield)
# Updated quarterly, used when real-time data unavailable
DEFAULT_DIVIDEND_YIELDS = {
    # ETFs
    "SPY": 0.0127,  # ~1.27%
    "QQQ": 0.0055,  # ~0.55%
    "IWM": 0.0133,  # ~1.33%
    "GDX": 0.0090,  # ~0.90%
    "XLE": 0.0350,  # ~3.50%
    "XLF": 0.0180,  # ~1.80%
    "XLI": 0.0150,  # ~1.50%
    "XLP": 0.0250,  # ~2.50%
    "XLV": 0.0160,  # ~1.60%
    "XLY": 0.0100,  # ~1.00%
    "XME": 0.0120,  # ~1.20%
    # Individual stocks
    "AAPL": 0.0050,  # ~0.50%
    "AAL": 0.0000,  # No dividend
    "AMD": 0.0000,  # No dividend
    "AMZN": 0.0000,  # No dividend
    "BA": 0.0000,  # Suspended dividend
    "BABA": 0.0000,  # No dividend
    "BAC": 0.0240,  # ~2.40%
    "GOOGL": 0.0050,  # ~0.50%
    "META": 0.0037,  # ~0.37%
    "MSFT": 0.0075,  # ~0.75%
    "NFLX": 0.0000,  # No dividend
    "NKE": 0.0130,  # ~1.30%
    "NVDA": 0.0003,  # ~0.03%
    "TSLA": 0.0000,  # No dividend
    "XOM": 0.0340,  # ~3.40%
}


def get_dividend_yield(symbol: str) -> float:
    """
    Get dividend yield for a symbol.

    Args:
        symbol: Ticker symbol

    Returns:
        Continuous dividend yield (annualized)
    """
    return DEFAULT_DIVIDEND_YIELDS.get(symbol.upper(), 0.0)


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    q: float = 0.0,
) -> float:
    """
    Calculate Black-Scholes option price with optional dividend adjustment.

    This is the Black-Scholes-Merton model that accounts for continuous
    dividend yield. When q=0, it reduces to standard Black-Scholes.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: CALL or PUT
        q: Continuous dividend yield (default 0)

    Returns:
        Option price
    """
    if T <= 0:
        # At expiry, return intrinsic value
        if option_type == OptionType.CALL:
            return max(0, S - K)
        else:
            return max(0, K - S)

    if sigma <= 0:
        raise ValueError("Volatility must be positive")

    # Black-Scholes-Merton with dividends
    # d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == OptionType.CALL:
        # C = S*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)
        price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(
            d2
        )
    else:
        # P = K*e^(-rT)*N(-d2) - S*e^(-qT)*N(-d1)
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(
            -d1
        )

    return price


def black_scholes_merton(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    dividend_yield: float,
) -> float:
    """
    Black-Scholes-Merton price with continuous dividend yield.

    Explicit alias for black_scholes_price with dividend yield.

    The Merton model adjusts for continuous dividends by:
    1. Reducing the forward price: F = S * exp((r-q)*T)
    2. Adjusting d1 and d2 accordingly

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: CALL or PUT
        dividend_yield: Continuous dividend yield (annualized)

    Returns:
        Option price
    """
    return black_scholes_price(S, K, T, r, sigma, option_type, q=dividend_yield)


def price_with_symbol(
    symbol: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    """
    Calculate option price using symbol's dividend yield.

    Convenience function that looks up dividend yield from defaults.

    Args:
        symbol: Ticker symbol
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: CALL or PUT

    Returns:
        Option price with dividend adjustment
    """
    q = get_dividend_yield(symbol)
    return black_scholes_price(S, K, T, r, sigma, option_type, q=q)


def black_scholes_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    q: float = 0.0,
) -> Greeks:
    """
    Calculate all Black-Scholes Greeks with dividend adjustment.

    Uses Black-Scholes-Merton formulas when q > 0.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: CALL or PUT
        q: Continuous dividend yield (default 0)

    Returns:
        Greeks dataclass with all values
    """
    if T <= 0 or sigma <= 0:
        # At expiry or zero vol, return limit values
        if option_type == OptionType.CALL:
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0

        return Greeks(
            delta=delta,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0,
        )

    sqrt_T = math.sqrt(T)
    # d1 with dividend yield
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Discount factors
    df_div = math.exp(-q * T)  # Dividend discount
    df_rate = math.exp(-r * T)  # Rate discount

    # Delta (with dividend adjustment)
    if option_type == OptionType.CALL:
        delta = df_div * norm.cdf(d1)
    else:
        delta = df_div * (norm.cdf(d1) - 1)

    # Gamma (same for calls and puts, with dividend)
    gamma = df_div * norm.pdf(d1) / (S * sigma * sqrt_T)

    # Theta (per day, with dividend)
    pdf_d1 = norm.pdf(d1)
    first_term = -(S * df_div * pdf_d1 * sigma) / (2 * sqrt_T)

    if option_type == OptionType.CALL:
        second_term = q * S * df_div * norm.cdf(d1)
        third_term = -r * K * df_rate * norm.cdf(d2)
        theta = (first_term + second_term + third_term) / 365  # Daily theta
    else:
        second_term = -q * S * df_div * norm.cdf(-d1)
        third_term = r * K * df_rate * norm.cdf(-d2)
        theta = (first_term + second_term + third_term) / 365  # Daily theta

    # Vega (per 1% move in volatility, with dividend)
    vega = S * df_div * sqrt_T * pdf_d1 / 100

    # Rho (per 1% move in rate)
    if option_type == OptionType.CALL:
        rho = K * T * df_rate * norm.cdf(d2) / 100
    else:
        rho = -K * T * df_rate * norm.cdf(-d2) / 100

    return Greeks(
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho,
    )


def greeks_with_symbol(
    symbol: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
) -> Greeks:
    """
    Calculate Greeks using symbol's dividend yield.

    Args:
        symbol: Ticker symbol
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: CALL or PUT

    Returns:
        Greeks with dividend adjustment
    """
    q = get_dividend_yield(symbol)
    return black_scholes_greeks(S, K, T, r, sigma, option_type, q=q)


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    q: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Optional[float]:
    """
    Calculate implied volatility using Brent's method with dividend adjustment.

    Args:
        market_price: Observed option price
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        option_type: CALL or PUT
        q: Continuous dividend yield (default 0)
        tol: Tolerance for convergence
        max_iter: Maximum iterations

    Returns:
        Implied volatility, or None if not found
    """
    if T <= 0:
        return None

    # Check for arbitrage violations (with dividend adjustment)
    forward_S = S * math.exp((r - q) * T)
    if option_type == OptionType.CALL:
        intrinsic = max(0, S * math.exp(-q * T) - K * math.exp(-r * T))
    else:
        intrinsic = max(0, K * math.exp(-r * T) - S * math.exp(-q * T))

    if market_price < intrinsic * 0.99:  # Allow small tolerance
        # Price below intrinsic - likely arbitrage
        return None

    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type, q=q) - market_price

    try:
        # Brent's method in range [0.01, 5.0] (1% to 500% vol)
        iv = brentq(objective, 0.01, 5.0, xtol=tol, maxiter=max_iter)
        return iv
    except (ValueError, RuntimeError):
        # Convergence failed
        return None


def implied_volatility_with_symbol(
    market_price: float,
    symbol: str,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
) -> Optional[float]:
    """
    Calculate implied volatility using symbol's dividend yield.

    Args:
        market_price: Observed option price
        symbol: Ticker symbol
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        option_type: CALL or PUT

    Returns:
        Implied volatility, or None if not found
    """
    q = get_dividend_yield(symbol)
    return implied_volatility(market_price, S, K, T, r, option_type, q=q)


def calculate_moneyness(
    S: float,
    K: float,
    option_type: OptionType,
) -> float:
    """
    Calculate moneyness (log-moneyness).

    Args:
        S: Stock price
        K: Strike price
        option_type: CALL or PUT

    Returns:
        Log-moneyness (positive = ITM for calls, OTM for puts)
    """
    return math.log(S / K)


def delta_to_strike(
    S: float,
    T: float,
    r: float,
    sigma: float,
    target_delta: float,
    option_type: OptionType,
    q: float = 0.0,
) -> float:
    """
    Find strike for a given delta with dividend adjustment.

    Args:
        S: Stock price
        T: Time to expiration
        r: Risk-free rate
        sigma: Volatility
        target_delta: Target delta (e.g., 0.40 for 40-delta call)
        option_type: CALL or PUT
        q: Continuous dividend yield (default 0)

    Returns:
        Strike price
    """
    if T <= 0 or sigma <= 0:
        return S  # ATM at expiry

    sqrt_T = math.sqrt(T)

    # Adjust target delta for dividend discount
    df_div = math.exp(-q * T)
    adjusted_delta = target_delta / df_div if df_div > 0 else target_delta

    if option_type == OptionType.CALL:
        # Delta_call = e^(-qT) * N(d1), so d1 = N^{-1}(adjusted_delta)
        d1 = norm.ppf(min(0.999, adjusted_delta))
    else:
        # Delta_put = e^(-qT) * (N(d1) - 1), so d1 = N^{-1}(adjusted_delta + 1)
        d1 = norm.ppf(max(0.001, adjusted_delta + 1))

    # Solve for K from d1 formula (with dividend)
    # d1 = (ln(S/K) + (r - q + σ²/2)T) / (σ√T)
    # K = S * exp(-(d1 * σ√T - (r - q + σ²/2)T))

    K = S * math.exp(-(d1 * sigma * sqrt_T - (r - q + 0.5 * sigma**2) * T))

    return K


def calculate_break_even(
    entry_price: float,
    strike: float,
    option_type: OptionType,
    is_long: bool = True,
) -> float:
    """
    Calculate breakeven price at expiration.

    Args:
        entry_price: Premium paid or received
        strike: Strike price
        option_type: CALL or PUT
        is_long: True if long, False if short

    Returns:
        Breakeven underlying price
    """
    if is_long:
        if option_type == OptionType.CALL:
            return strike + entry_price
        else:
            return strike - entry_price
    else:
        # Short position - breakeven is opposite
        if option_type == OptionType.CALL:
            return strike + entry_price
        else:
            return strike - entry_price


def calculate_payoff(
    S_final: float,
    K: float,
    premium: float,
    option_type: OptionType,
    quantity: int,
) -> float:
    """
    Calculate P&L at expiration.

    Args:
        S_final: Stock price at expiration
        K: Strike price
        premium: Entry premium (positive = paid, negative = received)
        option_type: CALL or PUT
        quantity: Number of contracts (positive = long, negative = short)

    Returns:
        Total P&L
    """
    if option_type == OptionType.CALL:
        intrinsic = max(0, S_final - K)
    else:
        intrinsic = max(0, K - S_final)

    pnl_per_contract = (intrinsic - premium) * 100
    return quantity * pnl_per_contract


def estimate_slippage(
    mid_price: float,
    bid: float,
    ask: float,
    volume: int,
    open_interest: int,
    is_buy: bool = True,
    base_spread_pct: float = 0.05,
) -> float:
    """
    Estimate slippage for options trade.

    Uses parametric model from plan:
    - Base spread percentage
    - Liquidity factor from volume/OI
    - OTM penalty (wider spreads for deep OTM)

    Args:
        mid_price: Mid price
        bid: Bid price
        ask: Ask price
        volume: Daily volume
        open_interest: Open interest
        is_buy: True if buying, False if selling
        base_spread_pct: Base spread percentage (default 5%)

    Returns:
        Estimated execution price
    """
    if mid_price <= 0:
        return mid_price

    # Actual spread
    actual_spread = ask - bid
    actual_spread_pct = actual_spread / mid_price if mid_price > 0 else base_spread_pct

    # Liquidity factor
    liquidity = volume + open_interest
    if liquidity > 0:
        liquidity_factor = 1 / math.log(1 + liquidity)
    else:
        liquidity_factor = 2.0  # Very illiquid

    # Estimated spread
    estimated_spread_pct = max(actual_spread_pct, base_spread_pct * liquidity_factor)

    # Execution price (cross half the spread)
    half_spread = mid_price * estimated_spread_pct / 2

    if is_buy:
        return mid_price + half_spread
    else:
        return mid_price - half_spread
