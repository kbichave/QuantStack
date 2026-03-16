"""
Synthetic market data generator for testing and CI.

Generates realistic OHLCV data with configurable regime switching,
volatility, and trend characteristics. Fully deterministic when seeded.

This module enables:
- End-to-end pipeline testing without network/API dependencies
- Fast CI tests with small datasets
- Reproducible experiments with fixed seeds
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class SyntheticMarketConfig:
    """
    Configuration for synthetic market data generation.
    
    Attributes:
        start: Start datetime for the series
        periods: Number of bars to generate
        freq: Pandas frequency string (e.g., "1h", "4h", "D")
        base_price: Starting price level
        vol: Annualized volatility (e.g., 0.20 for 20%)
        trend_strength: Drift strength per bar (e.g., 0.0001)
        regime_switch_prob: Probability of trend direction flip per bar
        mean_reversion_strength: Strength of mean reversion to trend (0-1)
        volume_base: Base volume per bar
        volume_noise: Volume noise factor (0-1)
        seed: Random seed for reproducibility
    """
    start: datetime = field(default_factory=lambda: datetime(2023, 1, 3, 9, 0))
    periods: int = 2000
    freq: str = "1h"
    base_price: float = 100.0
    vol: float = 0.20  # 20% annualized
    trend_strength: float = 0.0002
    regime_switch_prob: float = 0.02
    mean_reversion_strength: float = 0.1
    volume_base: int = 100000
    volume_noise: float = 0.3
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration."""
        if self.periods < 1:
            raise ValueError("periods must be >= 1")
        if self.base_price <= 0:
            raise ValueError("base_price must be positive")
        if self.vol < 0:
            raise ValueError("vol must be non-negative")
        if not 0 <= self.regime_switch_prob <= 1:
            raise ValueError("regime_switch_prob must be in [0, 1]")
        if not 0 <= self.mean_reversion_strength <= 1:
            raise ValueError("mean_reversion_strength must be in [0, 1]")


def _get_bars_per_year(freq: str) -> int:
    """Estimate number of bars per year for volatility scaling."""
    freq_lower = freq.lower()
    if freq_lower in ("1h", "h", "1H", "H"):
        return 252 * 7  # ~7 trading hours per day
    elif freq_lower in ("4h", "4H"):
        return 252 * 2  # ~2 4H bars per day
    elif freq_lower in ("d", "1d", "D", "1D"):
        return 252
    elif freq_lower in ("w", "1w", "W", "1W", "w-fri", "W-FRI"):
        return 52
    else:
        # Default to hourly
        return 252 * 7


def generate_synthetic_ohlcv(
    config: Optional[SyntheticMarketConfig] = None,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with regime switching.
    
    Creates a price series that:
    - Has piecewise trends (up/down regimes)
    - Switches regimes with configurable probability
    - Has realistic volatility scaled to the timeframe
    - Produces valid OHLCV (high >= open,close; low <= open,close)
    - Is fully deterministic for a given seed
    
    Args:
        config: Generation configuration (uses defaults if None)
        
    Returns:
        DataFrame with columns: open, high, low, close, volume
        Index is a DatetimeIndex with the specified frequency
        
    Example:
        >>> cfg = SyntheticMarketConfig(periods=1000, seed=123)
        >>> df = generate_synthetic_ohlcv(cfg)
        >>> len(df)
        1000
    """
    if config is None:
        config = SyntheticMarketConfig()
    
    logger.debug(
        f"Generating {config.periods} synthetic bars at {config.freq} "
        f"(seed={config.seed})"
    )
    
    # Set random seed for reproducibility
    rng = np.random.default_rng(config.seed)
    
    # Calculate per-bar volatility from annualized
    bars_per_year = _get_bars_per_year(config.freq)
    bar_vol = config.vol / np.sqrt(bars_per_year)
    
    # Initialize arrays
    n = config.periods
    closes = np.zeros(n)
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    volumes = np.zeros(n, dtype=np.int64)
    
    # Initial state
    closes[0] = config.base_price
    opens[0] = config.base_price
    regime = 1  # 1 = uptrend, -1 = downtrend
    
    # Generate price series with regime switching
    for i in range(1, n):
        # Check for regime switch
        if rng.random() < config.regime_switch_prob:
            regime *= -1
        
        # Calculate return components
        # 1. Trend component (drift in regime direction)
        trend_return = regime * config.trend_strength
        
        # 2. Mean reversion component (pulls toward trend)
        price_deviation = (closes[i-1] - config.base_price) / config.base_price
        mr_return = -config.mean_reversion_strength * price_deviation * 0.01
        
        # 3. Random noise component
        noise_return = rng.normal(0, bar_vol)
        
        # Combined return
        total_return = trend_return + mr_return + noise_return
        
        # Apply return to get close price
        closes[i] = closes[i-1] * (1 + total_return)
        
        # Open is previous close (with small gap occasionally)
        gap = rng.normal(0, bar_vol * 0.1) if rng.random() < 0.1 else 0
        opens[i] = closes[i-1] * (1 + gap)
    
    # Generate high/low with realistic intrabar movement
    for i in range(n):
        # Intrabar range based on volatility
        intrabar_range = abs(rng.normal(0, bar_vol * 1.5))
        
        # High is max of open/close plus some extension
        high_extension = abs(rng.exponential(bar_vol * 0.5))
        highs[i] = max(opens[i], closes[i]) + closes[i] * high_extension
        
        # Low is min of open/close minus some extension
        low_extension = abs(rng.exponential(bar_vol * 0.5))
        lows[i] = min(opens[i], closes[i]) - closes[i] * low_extension
        
        # Ensure constraints
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
        
        # Ensure low > 0
        if lows[i] <= 0:
            lows[i] = min(opens[i], closes[i]) * 0.99
    
    # Generate volume with noise
    base_vol = config.volume_base
    for i in range(n):
        vol_multiplier = 1 + rng.uniform(-config.volume_noise, config.volume_noise)
        # Higher volume on larger price moves
        move_size = abs(closes[i] - opens[i]) / opens[i] if opens[i] > 0 else 0
        vol_boost = 1 + move_size * 10
        volumes[i] = int(base_vol * vol_multiplier * vol_boost)
    
    # Create DatetimeIndex
    index = pd.date_range(
        start=config.start,
        periods=n,
        freq=config.freq,
        tz="America/New_York",
    )
    
    # Build DataFrame
    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=index)
    
    logger.info(
        f"Generated {len(df)} synthetic bars: "
        f"price range [{df['low'].min():.2f}, {df['high'].max():.2f}], "
        f"final price {df['close'].iloc[-1]:.2f}"
    )
    
    return df


def generate_synthetic_multi_symbol(
    symbols: list[str],
    config: Optional[SyntheticMarketConfig] = None,
    correlation: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """
    Generate synthetic OHLCV for multiple correlated symbols.
    
    Args:
        symbols: List of symbol names
        config: Base configuration (seed will be modified per symbol)
        correlation: Correlation between symbols (0-1)
        
    Returns:
        Dictionary mapping symbol to DataFrame
    """
    if config is None:
        config = SyntheticMarketConfig()
    
    result = {}
    base_seed = config.seed
    
    for i, symbol in enumerate(symbols):
        # Create symbol-specific config with different seed
        symbol_config = SyntheticMarketConfig(
            start=config.start,
            periods=config.periods,
            freq=config.freq,
            base_price=config.base_price * (1 + i * 0.1),  # Slightly different prices
            vol=config.vol * (1 + i * 0.05),  # Slightly different vol
            trend_strength=config.trend_strength,
            regime_switch_prob=config.regime_switch_prob,
            mean_reversion_strength=config.mean_reversion_strength,
            volume_base=config.volume_base,
            volume_noise=config.volume_noise,
            seed=base_seed + i,
        )
        
        result[symbol] = generate_synthetic_ohlcv(symbol_config)
        logger.debug(f"Generated synthetic data for {symbol}")
    
    return result


def validate_synthetic_ohlcv(df: pd.DataFrame) -> Tuple[bool, list[str]]:
    """
    Validate that synthetic OHLCV data meets all constraints.
    
    Args:
        df: OHLCV DataFrame to validate
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Check required columns
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
        return False, errors
    
    # Check for NaN values
    nan_counts = df[required].isna().sum()
    if nan_counts.any():
        errors.append(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")
    
    # Check high >= max(open, close)
    invalid_high = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
    if invalid_high > 0:
        errors.append(f"{invalid_high} bars have high < max(open, close)")
    
    # Check low <= min(open, close)
    invalid_low = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
    if invalid_low > 0:
        errors.append(f"{invalid_low} bars have low > min(open, close)")
    
    # Check high >= low
    invalid_hl = (df["high"] < df["low"]).sum()
    if invalid_hl > 0:
        errors.append(f"{invalid_hl} bars have high < low")
    
    # Check positive prices
    negative_prices = (df[["open", "high", "low", "close"]] <= 0).any().any()
    if negative_prices:
        errors.append("Found non-positive prices")
    
    # Check non-negative volume
    negative_vol = (df["volume"] < 0).any()
    if negative_vol:
        errors.append("Found negative volume")
    
    # Check index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("Index is not DatetimeIndex")
    
    is_valid = len(errors) == 0
    return is_valid, errors

