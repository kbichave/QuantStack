# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantCore MCP market tools — snapshots, regime, screening, trade templates, and calendars.

Extracted from ``quantcore.mcp.server`` to keep tool modules focused.
All helpers come from ``quantcore.mcp._helpers``; the ``mcp`` singleton
is imported from ``quantcore.mcp.server``.
"""

from typing import Any

import numpy as np
import pandas as pd

from quantcore.config.timeframes import Timeframe
from quantcore.mcp._helpers import _get_reader, _parse_timeframe
from quantcore.mcp.server import mcp

# =============================================================================
# MAS-ORIENTED TOOLS (Multi-Agent System Optimization)
# =============================================================================


@mcp.tool()
async def get_symbol_snapshot(
    symbol: str,
    timeframe: str = "daily",
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Get a unified snapshot of a symbol for fast MAS reasoning.

    Combines price, technical, volatility, and wave data in a single call
    to reduce agent reasoning overhead and tool call latency.

    Args:
        symbol: Stock/ETF symbol
        timeframe: Data timeframe ("daily", "1h", "4h")
        end_date: End date filter (YYYY-MM-DD) for historical simulation.
                  If provided, returns snapshot as of this date.

    Returns:
        Dictionary with:
            - price: Latest OHLCV
            - technicals: RSI, MACD, ATR, key MAs
            - volatility: Realized vol, IV rank (if available)
            - trend: Direction, strength, regime
            - levels: Support/resistance, key pivots
    """
    from quantcore.features.factory import MultiTimeframeFeatureFactory

    store = _get_reader()
    tf = _parse_timeframe(timeframe)

    try:
        df = store.load_ohlcv(symbol, tf)

        if df.empty:
            return {"error": f"No data for {symbol}", "symbol": symbol}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {
                    "error": f"No data for {symbol} before {end_date}",
                    "symbol": symbol,
                }

        # Get latest bar
        latest = df.iloc[-1]

        # Compute features
        factory = MultiTimeframeFeatureFactory(
            include_rrg=False,
            include_waves=False,
            include_technical_indicators=True,
        )
        features = factory.compute_all_timeframes({tf: df})[tf]
        latest_features = features.iloc[-1]

        # Extract key metrics
        snapshot = {
            "symbol": symbol,
            "timeframe": tf.value,
            "timestamp": str(df.index[-1]),
            "price": {
                "open": round(latest["open"], 2),
                "high": round(latest["high"], 2),
                "low": round(latest["low"], 2),
                "close": round(latest["close"], 2),
                "volume": int(latest["volume"]),
                "change_pct": (
                    round((latest["close"] / df["close"].iloc[-2] - 1) * 100, 2)
                    if len(df) > 1
                    else 0
                ),
            },
            "technicals": {},
            "volatility": {},
            "trend": {},
            "levels": {},
        }

        # Technical indicators
        for col in ["rsi_14", "macd", "macd_signal", "adx_14"]:
            if col in latest_features.index:
                snapshot["technicals"][col] = round(float(latest_features[col]), 2)

        # ATR
        if "atr_14" in latest_features.index:
            atr = float(latest_features["atr_14"])
            snapshot["technicals"]["atr"] = round(atr, 2)
            snapshot["technicals"]["atr_pct"] = round(atr / latest["close"] * 100, 2)

        # Volatility
        returns = df["close"].pct_change().dropna()
        if len(returns) > 20:
            realized_vol = float(returns.tail(20).std() * np.sqrt(252))
            snapshot["volatility"]["realized_vol_20d"] = round(realized_vol, 4)

        # Trend
        if "ema_20" in latest_features.index and "ema_50" in latest_features.index:
            ema_20 = float(latest_features["ema_20"])
            ema_50 = float(latest_features["ema_50"])
            snapshot["trend"]["ema_alignment"] = "bullish" if ema_20 > ema_50 else "bearish"
            snapshot["trend"]["above_ema_20"] = latest["close"] > ema_20

        # Support/Resistance (simple high/low levels)
        if len(df) > 20:
            snapshot["levels"]["high_20d"] = round(float(df["high"].tail(20).max()), 2)
            snapshot["levels"]["low_20d"] = round(float(df["low"].tail(20).min()), 2)

        return snapshot

    except Exception as e:
        return {"error": str(e), "symbol": symbol}
    finally:
        store.close()


@mcp.tool()
async def get_market_regime_snapshot(
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Get high-level market regime for global filters in MAS.

    Returns market-wide regime information useful for gating trades
    and adjusting strategy parameters across all agents.

    Args:
        end_date: End date filter (YYYY-MM-DD) for historical simulation.
                  If provided, returns regime as of this date.

    Returns:
        Dictionary with:
            - trend_regime: bull, bear, or sideways
            - volatility_regime: low, normal, or high (based on VIX proxy)
            - breadth: Market breadth indicator
            - risk_appetite: Risk-on or risk-off signal
    """
    from quantcore.features.factory import MultiTimeframeFeatureFactory
    from quantcore.hierarchy.regime_classifier import WeeklyRegimeClassifier

    store = _get_reader()

    try:
        # Use SPY as market proxy
        df = store.load_ohlcv("SPY", _parse_timeframe("daily"))

        if df.empty:
            return {"error": "No SPY data available for regime detection"}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {"error": f"No SPY data before {end_date}"}

        # Compute features
        factory = MultiTimeframeFeatureFactory(include_rrg=False)
        features = factory.compute_all_timeframes({_parse_timeframe("daily"): df})[
            _parse_timeframe("daily")
        ]

        # Classify regime
        classifier = WeeklyRegimeClassifier()
        regime_ctx = classifier.classify(features)

        # Calculate volatility regime
        returns = df["close"].pct_change().dropna()
        current_vol = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) > 20 else 0.15

        if current_vol < 0.12:
            vol_regime = "low"
        elif current_vol < 0.20:
            vol_regime = "normal"
        else:
            vol_regime = "high"

        # Trend detection
        latest = features.iloc[-1]
        close = df["close"].iloc[-1]

        ema_20 = float(latest.get("ema_20", close))
        ema_50 = float(latest.get("ema_50", close))

        if close > ema_20 > ema_50:
            trend = "bull"
        elif close < ema_20 < ema_50:
            trend = "bear"
        else:
            trend = "sideways"

        return {
            "timestamp": str(df.index[-1]),
            "market_proxy": "SPY",
            "regime": {
                "trend": trend,
                "volatility": vol_regime,
                "confidence": round(regime_ctx.confidence, 2),
            },
            "metrics": {
                "spy_price": round(close, 2),
                "realized_vol_20d": round(current_vol, 4),
                "ema_alignment": regime_ctx.ema_alignment,
                "momentum_score": round(regime_ctx.momentum_score, 2),
            },
            "signals": {
                "allows_long": regime_ctx.allows_long(),
                "allows_short": regime_ctx.allows_short(),
                "risk_appetite": (
                    "risk_on" if trend == "bull" and vol_regime != "high" else "risk_off"
                ),
            },
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()


@mcp.tool()
async def generate_trade_template(
    symbol: str,
    direction: str,
    structure_type: str = "vertical",
    expiry_days: int = 30,
    risk_amount: float = 500.0,
    underlying_price: float | None = None,
    iv_estimate: float = 0.25,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Generate a fully structured trade template for MAS approval workflow.

    Creates a complete trade specification that can be passed to
    HumanApprovalAgent or AutoExecutionAgent for final validation.

    Args:
        symbol: Underlying symbol
        direction: "bullish", "bearish", or "neutral"
        structure_type: "vertical", "single", "straddle", "iron_condor"
        expiry_days: Days to target expiration
        risk_amount: Maximum dollar risk for the trade
        underlying_price: Current price (fetched if not provided)
        iv_estimate: Estimated IV for pricing
        end_date: End date filter (YYYY-MM-DD) for historical simulation.
                  If provided, uses price as of this date.

    Returns:
        Dictionary with complete trade template:
            - legs: Fully specified option legs
            - risk: Max loss, max profit, break-evens
            - greeks: Position Greeks
            - validation_status: Pre-validation results
    """
    from quantcore.options.adapters.quantsbin_adapter import analyze_structure_quantsbin

    try:
        # Get current price if not provided
        if underlying_price is None:
            store = _get_reader()
            df = store.load_ohlcv(symbol, _parse_timeframe("daily"))
            store.close()

            if df.empty:
                return {"error": f"No price data for {symbol}"}

            # Filter to end_date if provided (for historical simulation)
            if end_date and not df.empty:
                end_dt = pd.to_datetime(end_date)
                df = df[df.index <= end_dt]
                if df.empty:
                    return {"error": f"No price data for {symbol} before {end_date}"}

            underlying_price = float(df["close"].iloc[-1])

        # Round to nearest strike increment
        if underlying_price > 100:
            strike_increment = 5.0
        elif underlying_price > 50:
            strike_increment = 2.5
        else:
            strike_increment = 1.0

        atm_strike = round(underlying_price / strike_increment) * strike_increment

        # Build legs based on structure type
        legs = []

        if structure_type == "single":
            opt_type = "call" if direction == "bullish" else "put"
            strike = atm_strike if direction != "neutral" else atm_strike

            legs.append(
                {
                    "option_type": opt_type,
                    "strike": strike,
                    "expiry_days": expiry_days,
                    "quantity": 1,
                    "iv": iv_estimate,
                }
            )

        elif structure_type == "vertical":
            if direction == "bullish":
                legs = [
                    {
                        "option_type": "call",
                        "strike": atm_strike,
                        "expiry_days": expiry_days,
                        "quantity": 1,
                        "iv": iv_estimate,
                    },
                    {
                        "option_type": "call",
                        "strike": atm_strike + strike_increment * 2,
                        "expiry_days": expiry_days,
                        "quantity": -1,
                        "iv": iv_estimate * 0.95,
                    },
                ]
            elif direction == "bearish":
                legs = [
                    {
                        "option_type": "put",
                        "strike": atm_strike,
                        "expiry_days": expiry_days,
                        "quantity": 1,
                        "iv": iv_estimate,
                    },
                    {
                        "option_type": "put",
                        "strike": atm_strike - strike_increment * 2,
                        "expiry_days": expiry_days,
                        "quantity": -1,
                        "iv": iv_estimate * 0.95,
                    },
                ]
            else:  # neutral - iron butterfly
                legs = [
                    {
                        "option_type": "put",
                        "strike": atm_strike,
                        "expiry_days": expiry_days,
                        "quantity": -1,
                        "iv": iv_estimate,
                    },
                    {
                        "option_type": "call",
                        "strike": atm_strike,
                        "expiry_days": expiry_days,
                        "quantity": -1,
                        "iv": iv_estimate,
                    },
                ]

        elif structure_type == "straddle":
            legs = [
                {
                    "option_type": "call",
                    "strike": atm_strike,
                    "expiry_days": expiry_days,
                    "quantity": 1 if direction != "neutral" else -1,
                    "iv": iv_estimate,
                },
                {
                    "option_type": "put",
                    "strike": atm_strike,
                    "expiry_days": expiry_days,
                    "quantity": 1 if direction != "neutral" else -1,
                    "iv": iv_estimate,
                },
            ]

        elif structure_type == "iron_condor":
            width = strike_increment * 2
            legs = [
                {
                    "option_type": "put",
                    "strike": atm_strike - width * 2,
                    "expiry_days": expiry_days,
                    "quantity": 1,
                    "iv": iv_estimate * 1.1,
                },
                {
                    "option_type": "put",
                    "strike": atm_strike - width,
                    "expiry_days": expiry_days,
                    "quantity": -1,
                    "iv": iv_estimate * 1.05,
                },
                {
                    "option_type": "call",
                    "strike": atm_strike + width,
                    "expiry_days": expiry_days,
                    "quantity": -1,
                    "iv": iv_estimate * 0.95,
                },
                {
                    "option_type": "call",
                    "strike": atm_strike + width * 2,
                    "expiry_days": expiry_days,
                    "quantity": 1,
                    "iv": iv_estimate * 0.9,
                },
            ]

        # Analyze structure
        structure_spec = {
            "underlying_symbol": symbol,
            "underlying_price": underlying_price,
            "legs": legs,
        }

        analysis = analyze_structure_quantsbin(structure_spec)

        # Calculate quantity based on risk
        if analysis.get("max_loss") and analysis["max_loss"] < 0:
            max_loss_per_contract = abs(analysis["max_loss"])
            quantity = max(1, int(risk_amount / max_loss_per_contract))
        else:
            quantity = 1

        # Scale legs by quantity
        for leg in legs:
            leg["quantity"] = leg["quantity"] * quantity

        # Recalculate with scaled quantity
        structure_spec["legs"] = legs
        analysis = analyze_structure_quantsbin(structure_spec)

        return {
            "template_id": f"{symbol}_{structure_type}_{direction}_{expiry_days}d",
            "symbol": symbol,
            "direction": direction,
            "structure_type": analysis.get("structure_type", structure_type),
            "underlying_price": underlying_price,
            "legs": legs,
            "risk_profile": {
                "max_profit": analysis.get("max_profit", 0),
                "max_loss": analysis.get("max_loss", 0),
                "break_evens": analysis.get("break_evens", []),
                "risk_reward_ratio": analysis.get("risk_reward_ratio", 0),
                "probability_of_profit": analysis.get("probability_of_profit"),
            },
            "greeks": analysis.get("greeks", {}),
            "validation": {
                "is_defined_risk": analysis.get("is_defined_risk", False),
                "within_risk_limit": abs(analysis.get("max_loss", 0)) <= risk_amount * 1.1,
                "has_positive_expectancy": analysis.get("max_profit", 0)
                > abs(analysis.get("max_loss", 0)) * 0.3,
            },
            "execution_notes": {
                "order_type": "limit",
                "price_target": (
                    round(analysis.get("total_premium", 0) / 100 / quantity, 2)
                    if quantity > 0
                    else 0
                ),
                "time_in_force": "day",
            },
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def validate_trade(
    trade_template: dict[str, Any],
    account_equity: float = 100000.0,
    max_position_pct: float = 5.0,
    max_daily_loss_pct: float = 2.0,
    current_daily_pnl: float = 0.0,
) -> dict[str, Any]:
    """
    Validate a trade template against risk rules before execution.

    Final gate before sending orders to broker. Checks position limits,
    risk constraints, and structure validity.

    Args:
        trade_template: Trade template from generate_trade_template
        account_equity: Total account equity
        max_position_pct: Maximum position size as % of equity
        max_daily_loss_pct: Maximum daily loss allowed as % of equity
        current_daily_pnl: Current day's realized P&L

    Returns:
        Dictionary with validation results:
            - is_valid: Boolean pass/fail
            - checks: Individual check results
            - warnings: Non-blocking concerns
            - rejection_reasons: If invalid, why
    """
    try:
        checks = {}
        warnings = []
        rejection_reasons = []

        # Extract key metrics
        max_loss = abs(trade_template.get("risk_profile", {}).get("max_loss", float("inf")))
        is_defined_risk = trade_template.get("validation", {}).get("is_defined_risk", False)
        symbol = trade_template.get("symbol", "UNKNOWN")

        # Check 1: Defined risk
        checks["defined_risk"] = is_defined_risk
        if not is_defined_risk:
            rejection_reasons.append("Trade has undefined/unlimited risk")

        # Check 2: Position size limit
        max_position_value = account_equity * max_position_pct / 100
        checks["within_position_limit"] = max_loss <= max_position_value
        if not checks["within_position_limit"]:
            rejection_reasons.append(
                f"Max loss ${max_loss:.0f} exceeds position limit ${max_position_value:.0f}"
            )

        # Check 3: Daily loss limit
        remaining_daily_risk = account_equity * max_daily_loss_pct / 100 + current_daily_pnl
        checks["within_daily_limit"] = max_loss <= remaining_daily_risk
        if not checks["within_daily_limit"]:
            rejection_reasons.append(
                f"Max loss would exceed daily limit. Remaining: ${remaining_daily_risk:.0f}"
            )

        # Check 4: Greeks sanity
        greeks = trade_template.get("greeks", {})
        delta = abs(greeks.get("delta", 0))

        checks["delta_reasonable"] = delta < 500  # Max 500 delta per position
        if not checks["delta_reasonable"]:
            rejection_reasons.append(f"Delta exposure {delta:.0f} too high")

        # Check 5: Legs consistency
        legs = trade_template.get("legs", [])
        checks["has_legs"] = len(legs) > 0
        if not checks["has_legs"]:
            rejection_reasons.append("No legs in trade template")

        # Check 6: Break-evens exist for spreads
        break_evens = trade_template.get("risk_profile", {}).get("break_evens", [])
        if len(legs) > 1 and len(break_evens) == 0:
            warnings.append("Multi-leg structure has no calculated break-evens")

        # Check 7: Risk/reward sanity
        rr_ratio = trade_template.get("risk_profile", {}).get("risk_reward_ratio", 0)
        if rr_ratio < 0.2:
            warnings.append(f"Risk/reward ratio {rr_ratio:.2f} is unfavorable")

        # Check 8: Expiration not too close
        legs_expiry = [leg.get("expiry_days", 30) for leg in legs]
        min_expiry = min(legs_expiry) if legs_expiry else 30
        checks["sufficient_time"] = min_expiry >= 3
        if not checks["sufficient_time"]:
            warnings.append(f"Expiration in {min_expiry} days - consider rolling")

        # Overall validation
        is_valid = len(rejection_reasons) == 0

        return {
            "is_valid": is_valid,
            "symbol": symbol,
            "checks": checks,
            "warnings": warnings,
            "rejection_reasons": rejection_reasons,
            "risk_summary": {
                "max_loss": max_loss,
                "max_position_allowed": max_position_value,
                "remaining_daily_risk": remaining_daily_risk,
                "delta_exposure": delta,
            },
            "approval_status": "APPROVED" if is_valid else "REJECTED",
        }

    except Exception as e:
        return {
            "is_valid": False,
            "error": str(e),
            "approval_status": "ERROR",
        }


# =============================================================================
# SCREENER TOOLS
# =============================================================================


@mcp.tool()
async def run_screener(
    symbols: list[str] | None = None,
    min_price: float = 10.0,
    max_price: float = 500.0,
    min_volume: int = 100000,
    trend_filter: str | None = None,
    rsi_oversold: float | None = None,
    rsi_overbought: float | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Run multi-factor screener across symbols.

    Filters symbols by price, volume, trend, and technical conditions.
    Used by OpportunityDiscoveryAgent to find trade candidates.

    Args:
        symbols: List of symbols to screen (uses config defaults if None)
        min_price: Minimum price filter
        max_price: Maximum price filter
        min_volume: Minimum average volume
        trend_filter: "bullish", "bearish", or None
        rsi_oversold: RSI below this value = oversold
        rsi_overbought: RSI above this value = overbought
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        Dictionary with matching symbols and their key metrics
    """
    from quantcore.config.settings import get_settings
    from quantcore.features.technical_indicators import TechnicalIndicators

    store = _get_reader()
    tf = _parse_timeframe("daily")
    end_dt = pd.to_datetime(end_date) if end_date else None

    try:
        # Get symbols to screen
        if symbols is None:
            settings = get_settings()
            symbols = settings.symbols[:20]  # Limit for performance

        matches = []

        for symbol in symbols:
            try:
                df = store.load_ohlcv(symbol, tf)

                # Filter to end_date if provided (for historical simulation)
                if end_dt is not None and not df.empty:
                    df = df[df.index <= end_dt]

                if df.empty or len(df) < 50:
                    continue

                latest = df.iloc[-1]
                price = float(latest["close"])
                volume = float(df["volume"].tail(20).mean())

                # Price filter
                if price < min_price or price > max_price:
                    continue

                # Volume filter
                if volume < min_volume:
                    continue

                # Compute indicators
                tech = TechnicalIndicators(tf)
                features = tech.compute(df)
                latest_features = features.iloc[-1]

                # Trend filter
                if trend_filter:
                    ema_20 = float(latest_features.get("ema_20", price))
                    ema_50 = float(latest_features.get("ema_50", price))

                    is_bullish = price > ema_20 > ema_50
                    is_bearish = price < ema_20 < ema_50

                    if trend_filter == "bullish" and not is_bullish:
                        continue
                    if trend_filter == "bearish" and not is_bearish:
                        continue

                # RSI filter
                rsi = float(latest_features.get("rsi_14", 50))

                if rsi_oversold and rsi > rsi_oversold:
                    continue
                if rsi_overbought and rsi < rsi_overbought:
                    continue

                # Add to matches
                matches.append(
                    {
                        "symbol": symbol,
                        "price": round(price, 2),
                        "volume": int(volume),
                        "rsi": round(rsi, 1),
                        "trend": "bullish" if price > ema_20 else "bearish",
                        "change_1d": (
                            round((price / df["close"].iloc[-2] - 1) * 100, 2) if len(df) > 1 else 0
                        ),
                    }
                )

            except Exception:
                continue

        # Sort by volume
        matches.sort(key=lambda x: x["volume"], reverse=True)

        return {
            "total_screened": len(symbols),
            "matches": len(matches),
            "filters_applied": {
                "price_range": [min_price, max_price],
                "min_volume": min_volume,
                "trend": trend_filter,
                "rsi_oversold": rsi_oversold,
                "rsi_overbought": rsi_overbought,
            },
            "results": matches[:20],  # Top 20
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()


# =============================================================================
# MICROSTRUCTURE TOOLS
# =============================================================================


@mcp.tool()
async def analyze_liquidity(
    symbol: str,
    timeframe: str = "daily",
    window: int = 20,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Analyze liquidity characteristics of a symbol.

    Computes:
    - Bid-ask spread estimates (Corwin-Schultz, Roll)
    - Volume analysis vs historical average
    - Liquidity score (0-1)

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        window: Lookback window for calculations
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        LiquidityFeatures with spread estimates and scores
    """
    from quantcore.microstructure.liquidity import LiquidityAnalyzer

    store = _get_reader()
    tf = _parse_timeframe(timeframe)

    try:
        df = store.load_ohlcv(symbol, tf)

        if df.empty:
            return {"error": f"No data for {symbol}"}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {"error": f"No data for {symbol} before {end_date}"}

        if len(df) < window + 10:
            return {"error": f"Need at least {window + 10} bars"}

        # Analyze liquidity
        analyzer = LiquidityAnalyzer(spread_threshold_bps=30, min_volume_ratio=0.5)
        features = analyzer.analyze(df, window=window)

        # Get latest features
        latest = features.iloc[-1] if not features.empty else None

        if latest is None:
            return {"error": "Could not compute liquidity features"}

        return {
            "symbol": symbol,
            "timeframe": tf.value,
            "window": window,
            "timestamp": str(df.index[-1]),
            "spread_estimates": {
                "corwin_schultz_bps": round(float(latest.get("cs_spread_bps", 0)), 2),
                "roll_spread_bps": round(float(latest.get("roll_spread_bps", 0)), 2),
                "combined_spread_bps": round(float(latest.get("estimated_spread_bps", 0)), 2),
            },
            "volume": {
                "current": int(df["volume"].iloc[-1]),
                "avg_20d": int(df["volume"].rolling(20).mean().iloc[-1]),
                "vs_avg_ratio": round(float(latest.get("volume_vs_avg", 1)), 2),
            },
            "liquidity_score": round(float(latest.get("liquidity_score", 0.5)), 2),
            "is_liquid": bool(latest.get("is_liquid", True)),
            "recommendations": [
                ("Liquid" if latest.get("is_liquid", True) else "Low liquidity - widen stops"),
                f"Estimated round-trip cost: {latest.get('estimated_spread_bps', 0) * 2:.1f} bps",
            ],
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()


@mcp.tool()
async def analyze_volume_profile(
    symbol: str,
    timeframe: str = "daily",
    lookback_days: int = 20,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Analyze volume profile and VWAP levels.

    Identifies:
    - Volume-weighted average price (VWAP)
    - High volume nodes (support/resistance)
    - Intraday volume patterns

    Args:
        symbol: Stock symbol
        timeframe: Data timeframe
        lookback_days: Days to analyze
        end_date: End date filter (YYYY-MM-DD) for historical simulation.

    Returns:
        Volume profile with VWAP levels and key nodes
    """
    store = _get_reader()
    tf = _parse_timeframe(timeframe)

    try:
        df = store.load_ohlcv(symbol, tf)

        if df.empty:
            return {"error": f"No data for {symbol}"}

        # Filter to end_date if provided (for historical simulation)
        if end_date and not df.empty:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
            if df.empty:
                return {"error": f"No data for {symbol} before {end_date}"}

        # Use last N days
        df = df.tail(lookback_days * (1 if tf == Timeframe.D1 else 7))

        if len(df) < 10:
            return {"error": "Insufficient data for volume profile"}

        # Calculate VWAP
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()

        current_price = float(df["close"].iloc[-1])
        current_vwap = float(vwap.iloc[-1])

        # Volume by price analysis (simplified)
        price_range = df["high"].max() - df["low"].min()
        n_bins = 10
        bin_size = price_range / n_bins

        volume_nodes = []
        for i in range(n_bins):
            bin_low = df["low"].min() + i * bin_size
            bin_high = bin_low + bin_size
            mask = (df["close"] >= bin_low) & (df["close"] < bin_high)
            vol = df.loc[mask, "volume"].sum()
            volume_nodes.append(
                {
                    "price_low": round(bin_low, 2),
                    "price_high": round(bin_high, 2),
                    "volume": int(vol),
                }
            )

        # Find high volume nodes (potential S/R)
        volume_nodes.sort(key=lambda x: x["volume"], reverse=True)
        high_volume_levels = volume_nodes[:3]

        return {
            "symbol": symbol,
            "timeframe": tf.value,
            "lookback_days": lookback_days,
            "vwap": {
                "current": round(current_vwap, 2),
                "price_vs_vwap_pct": round((current_price / current_vwap - 1) * 100, 2),
                "above_vwap": current_price > current_vwap,
            },
            "high_volume_nodes": high_volume_levels,
            "volume_stats": {
                "total_volume": int(df["volume"].sum()),
                "avg_daily_volume": int(df["volume"].mean()),
                "highest_volume_day": (
                    str(df["volume"].idxmax().date())
                    if hasattr(df["volume"].idxmax(), "date")
                    else str(df["volume"].idxmax())
                ),
            },
            "current_price": current_price,
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        store.close()


# =============================================================================
# CALENDAR TOOLS
# =============================================================================


@mcp.tool()
async def get_trading_calendar(
    year: int | None = None,
    month: int | None = None,
) -> dict[str, Any]:
    """
    Get market trading calendar information.

    Returns:
    - Market hours
    - Upcoming holidays
    - Early close days
    - Current market status

    Args:
        year: Year to query (default: current year)
        month: Month to query (default: current month)

    Returns:
        Calendar with holidays and trading hours
    """
    from datetime import datetime
    from datetime import time as dt_time

    try:
        now = datetime.now()
        year = year or now.year
        month = month or now.month

        # Standard US market hours
        market_hours = {
            "regular_open": "09:30",
            "regular_close": "16:00",
            "pre_market_open": "04:00",
            "pre_market_close": "09:30",
            "after_hours_open": "16:00",
            "after_hours_close": "20:00",
            "timezone": "America/New_York",
        }

        # US market holidays 2024-2025
        holidays = {
            2024: [
                ("2024-01-01", "New Year's Day"),
                ("2024-01-15", "MLK Day"),
                ("2024-02-19", "Presidents Day"),
                ("2024-03-29", "Good Friday"),
                ("2024-05-27", "Memorial Day"),
                ("2024-06-19", "Juneteenth"),
                ("2024-07-04", "Independence Day"),
                ("2024-09-02", "Labor Day"),
                ("2024-11-28", "Thanksgiving"),
                ("2024-12-25", "Christmas"),
            ],
            2025: [
                ("2025-01-01", "New Year's Day"),
                ("2025-01-20", "MLK Day"),
                ("2025-02-17", "Presidents Day"),
                ("2025-04-18", "Good Friday"),
                ("2025-05-26", "Memorial Day"),
                ("2025-06-19", "Juneteenth"),
                ("2025-07-04", "Independence Day"),
                ("2025-09-01", "Labor Day"),
                ("2025-11-27", "Thanksgiving"),
                ("2025-12-25", "Christmas"),
            ],
        }

        # Early close days (1pm ET)
        early_closes = {
            2024: [
                ("2024-07-03", "Day before Independence Day"),
                ("2024-11-29", "Day after Thanksgiving"),
                ("2024-12-24", "Christmas Eve"),
            ],
            2025: [
                ("2025-07-03", "Day before Independence Day"),
                ("2025-11-28", "Day after Thanksgiving"),
                ("2025-12-24", "Christmas Eve"),
            ],
        }

        # Check if market is open now (simplified)
        is_weekday = now.weekday() < 5
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        current_time = now.time()

        is_market_hours = is_weekday and market_open <= current_time <= market_close

        return {
            "query": {"year": year, "month": month},
            "market_hours": market_hours,
            "holidays": holidays.get(year, []),
            "early_closes": early_closes.get(year, []),
            "current_status": {
                "is_market_hours": is_market_hours,
                "current_time": str(now),
                "next_open": "09:30 ET" if not is_market_hours else "Market is open",
            },
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def get_event_calendar(
    symbol: str | None = None,
    days_ahead: int = 14,
) -> dict[str, Any]:
    """
    Get upcoming economic and earnings events.

    Returns events that may affect trading:
    - FOMC meetings
    - CPI/PPI releases
    - NFP (Non-Farm Payrolls)
    - Earnings dates (if symbol provided)
    - Options expiration

    Args:
        symbol: Stock symbol for earnings (optional)
        days_ahead: Days to look ahead

    Returns:
        Calendar of upcoming events with impact ratings
    """
    from datetime import datetime, timedelta

    try:
        now = datetime.now()
        end_date = now + timedelta(days=days_ahead)

        # Static economic calendar (would be fetched from API in production)
        # These are example dates for illustration
        economic_events = [
            {
                "date": "2024-12-18",
                "event": "FOMC Decision",
                "type": "FOMC",
                "impact": "HIGH",
            },
            {
                "date": "2024-12-20",
                "event": "PCE Inflation",
                "type": "INFLATION",
                "impact": "HIGH",
            },
            {
                "date": "2025-01-10",
                "event": "NFP Report",
                "type": "NFP",
                "impact": "HIGH",
            },
            {
                "date": "2025-01-15",
                "event": "CPI Release",
                "type": "CPI",
                "impact": "HIGH",
            },
            {
                "date": "2025-01-29",
                "event": "FOMC Decision",
                "type": "FOMC",
                "impact": "HIGH",
            },
        ]

        # Options expiration dates
        opex_events = []
        for month_offset in range(2):
            # Third Friday of each month
            first_day = (
                datetime(now.year, now.month + month_offset, 1)
                if now.month + month_offset <= 12
                else datetime(now.year + 1, (now.month + month_offset) % 12, 1)
            )
            # Find third Friday
            day = 1
            fridays = 0
            while fridays < 3:
                test_date = first_day.replace(day=day)
                if test_date.weekday() == 4:  # Friday
                    fridays += 1
                    if fridays == 3:
                        opex_events.append(
                            {
                                "date": test_date.strftime("%Y-%m-%d"),
                                "event": "Monthly Options Expiration",
                                "type": "OPEX",
                                "impact": "MEDIUM",
                            }
                        )
                day += 1

        # Combine and filter by date range
        all_events = economic_events + opex_events
        filtered_events = [
            e
            for e in all_events
            if now.strftime("%Y-%m-%d") <= e["date"] <= end_date.strftime("%Y-%m-%d")
        ]

        # Sort by date
        filtered_events.sort(key=lambda x: x["date"])

        result = {
            "query": {
                "symbol": symbol,
                "days_ahead": days_ahead,
                "start_date": now.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
            },
            "economic_events": filtered_events,
            "event_count": len(filtered_events),
            "high_impact_count": len([e for e in filtered_events if e["impact"] == "HIGH"]),
        }

        # Add blackout recommendations
        blackout_dates = [e["date"] for e in filtered_events if e["impact"] == "HIGH"]
        result["blackout_dates"] = blackout_dates
        result["recommendation"] = (
            "Avoid new positions on blackout dates"
            if blackout_dates
            else "No high-impact events in range"
        )

        return result

    except Exception as e:
        return {"error": str(e)}
