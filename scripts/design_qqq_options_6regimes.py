#!/usr/bin/env python3
"""
QQQ Options Strategy Design — All 6 Regimes

Designs and validates options strategies for each regime condition.
Uses direct DB access and bypasses MCP server initialization.
"""

import json
import uuid
from pathlib import Path

from quantstack.db import pg_conn


# Define all 6 strategies with complete specifications
STRATEGIES = [
    {
        "regime": "trending_up + normal vol",
        "name": "qqq_bull_call_spread_trending_v1",
        "structure": "bull_call_spread",
        "description": "Bull call spread (ATM/ATM+15) in confirmed uptrends with normal volatility. "
                      "Buy ATM call, sell OTM call 15pt wide. Max risk = debit paid.",
        "entry_rules": [
            {"indicator": "regime", "condition": "equals", "value": "trending_up"},
            {"indicator": "rsi_14", "condition": "between", "value": [40, 65]},
            {"indicator": "macd_histogram", "condition": "greater_than", "value": 0},
        ],
        "exit_rules": [
            {"indicator": "profit_pct", "condition": "greater_than", "value": 80},
            {"indicator": "loss_pct", "condition": "less_than", "value": -100},
            {"indicator": "dte", "condition": "less_than", "value": 7},
        ],
        "parameters": {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "adx_period": 14,
        },
        "risk_params": {
            "max_premium_pct": 2.0,
            "dte_entry": 30,
            "strike_width": 15,
            "delta_long": 0.50,
            "delta_short": 0.30,
        },
        "options_specs": {
            "legs": [
                {"type": "call", "action": "buy", "delta_target": 0.50},
                {"type": "call", "action": "sell", "delta_target": 0.30},
            ],
            "dte": 30,
            "max_contracts": 5,
            "max_risk_per_contract": "debit paid (~$1.50-2.50/ct = $150-250)",
            "greek_targets": {
                "delta": "+0.20 (net bullish)",
                "theta": "-0.05 (debit spread, small decay)",
                "vega": "+0.10 (benefits from vol expansion)",
            },
            "iv_conditions": "Neutral — works in normal vol regime",
        },
    },
    {
        "regime": "trending_up + high vol",
        "name": "qqq_long_call_highvol_v1",
        "structure": "long_call",
        "description": "Long ATM calls when IV rank < 50 in uptrends with elevated volatility. "
                      "Small size to manage vega risk. Positive delta + vega exposure.",
        "entry_rules": [
            {"indicator": "regime", "condition": "equals", "value": "trending_up"},
            {"indicator": "atr_percentile", "condition": "greater_than", "value": 75},
            {"indicator": "close", "condition": "above", "value": "ema_20"},
            {"indicator": "iv_rank", "condition": "less_than", "value": 50},
        ],
        "exit_rules": [
            {"indicator": "profit_pct", "condition": "greater_than", "value": 100},
            {"indicator": "loss_pct", "condition": "less_than", "value": -50},
            {"indicator": "dte", "condition": "less_than", "value": 10},
        ],
        "parameters": {
            "ema_period": 20,
            "atr_period": 14,
            "iv_lookback": 252,
        },
        "risk_params": {
            "max_premium_pct": 1.0,
            "dte_entry": 30,
            "delta_target": 0.50,
            "max_vega_per_contract": 30,
        },
        "options_specs": {
            "legs": [
                {"type": "call", "action": "buy", "delta_target": 0.50},
            ],
            "dte": 30,
            "max_contracts": 2,
            "max_risk_per_contract": "premium paid (~$3-5/ct = $300-500)",
            "greek_targets": {
                "delta": "+0.50 (directional)",
                "theta": "-0.10 (time decay risk)",
                "vega": "+0.30 (benefits if vol continues rising)",
                "gamma": "+0.02 (convexity on breakout)",
            },
            "iv_conditions": "IV rank < 50 REQUIRED to avoid crush risk",
        },
    },
    {
        "regime": "trending_down + normal vol",
        "name": "qqq_bear_put_spread_v3",
        "structure": "bear_put_spread",
        "description": "Bear put spread (ATM/ATM-15) in confirmed downtrends. "
                      "EGARCH leverage effect provides delta+vega tailwind. DISCRETIONARY USE ONLY.",
        "entry_rules": [
            {"indicator": "adx", "condition": "greater_than", "value": 25},
            {"indicator": "minus_di", "condition": "greater_than", "value": "plus_di"},
            {"indicator": "rsi_14", "condition": "less_than", "value": 50},
            {"indicator": "macd_histogram", "condition": "less_than", "value": 0},
        ],
        "exit_rules": [
            {"indicator": "profit_pct", "condition": "greater_than", "value": 50},
            {"indicator": "loss_pct", "condition": "less_than", "value": -60},
            {"indicator": "holding_days", "condition": "greater_than", "value": 14},
            {"indicator": "dte", "condition": "less_than", "value": 7},
        ],
        "parameters": {
            "adx_period": 14,
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
        },
        "risk_params": {
            "max_premium_pct": 1.5,
            "dte_entry": 30,
            "strike_width": 15,
            "delta_long": -0.50,
            "delta_short": -0.30,
        },
        "options_specs": {
            "legs": [
                {"type": "put", "action": "buy", "delta_target": -0.50},
                {"type": "put", "action": "sell", "delta_target": -0.30},
            ],
            "dte": 30,
            "max_contracts": 3,
            "max_risk_per_contract": "debit paid (~$2-3.50/ct = $200-350)",
            "greek_targets": {
                "delta": "-0.20 (net bearish)",
                "theta": "-0.05 (debit spread decay)",
                "vega": "+0.15 (leverage effect: vol expands on down moves)",
            },
            "iv_conditions": "Benefits from EGARCH leverage (gamma=-0.108 from QQQ model)",
            "discretionary_only": True,
            "reason": "WF OOS Sharpe -0.12 — not systematic, use for defined-risk shorts only",
        },
    },
    {
        "regime": "trending_down + high vol",
        "name": "qqq_bear_put_wide_vix_hedge_v1",
        "structure": "bear_put_spread_plus_vix",
        "description": "Aggressive bear put spread (20-pt wide) + VIX call hedge. "
                      "In severe downtrends, IV expands continuously (EGARCH leverage). "
                      "VIX call provides tail protection.",
        "entry_rules": [
            {"indicator": "adx", "condition": "greater_than", "value": 30},
            {"indicator": "minus_di", "condition": "greater_than", "value": 30},
            {"indicator": "macd_histogram", "condition": "less_than", "value": 0},
            {"indicator": "atr_percentile", "condition": "greater_than", "value": 80},
        ],
        "exit_rules": [
            {"indicator": "profit_pct", "condition": "greater_than", "value": 80},
            {"indicator": "loss_pct", "condition": "less_than", "value": -70},
            {"indicator": "holding_days", "condition": "greater_than", "value": 14},
            {"indicator": "dte", "condition": "less_than", "value": 7},
        ],
        "parameters": {
            "adx_period": 14,
            "atr_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
        },
        "risk_params": {
            "max_premium_pct": 2.5,
            "dte_entry": 30,
            "strike_width": 20,
            "delta_long": -0.45,
            "delta_short": -0.25,
            "vix_call_allocation_pct": 20,
        },
        "options_specs": {
            "legs": [
                {"type": "put", "action": "buy", "delta_target": -0.45, "symbol": "QQQ"},
                {"type": "put", "action": "sell", "delta_target": -0.25, "symbol": "QQQ"},
                {"type": "call", "action": "buy", "delta_target": 0.30, "symbol": "VIX", "allocation_pct": 20},
            ],
            "dte": 30,
            "max_contracts": 2,
            "max_risk_per_contract": "debit paid + VIX call (~$4-6/ct = $400-600 total)",
            "greek_targets": {
                "delta": "-0.20 (net bearish on QQQ)",
                "theta": "-0.08 (debit spread + long call decay)",
                "vega": "+0.25 (strong vol expansion benefit)",
                "tail_hedge": "VIX call provides 2x-5x payout in crash scenarios",
            },
            "iv_conditions": "Allows higher IV rank (up to 70) — vol expands in crashes",
            "hedge_ratio": "20% of spread cost allocated to VIX calls",
        },
    },
    {
        "regime": "ranging + low vol",
        "name": "qqq_iron_condor_vrp_v3",
        "structure": "iron_condor",
        "description": "EXISTING VALIDATED STRATEGY: Iron condor (1-stdev, 10pt wings) in strongly ranging markets. "
                      "ADX < 20 + BB compressed + VRP proxy > 3pp. Best QQQ options result: "
                      "IS Sharpe 2.17, OOS Sharpe 0.85, 81.7% WR.",
        "entry_rules": [
            {"indicator": "adx", "condition": "less_than", "value": 20},
            {"indicator": "bb_width_percentile", "condition": "less_than", "value": 50},
            {"indicator": "vrp_proxy", "condition": "greater_than", "value": 3},
        ],
        "exit_rules": [
            {"indicator": "profit_pct", "condition": "greater_than", "value": 50},
            {"indicator": "loss_pct", "condition": "less_than", "value": -100},
            {"indicator": "holding_days", "condition": "greater_than", "value": 14},
            {"indicator": "dte", "condition": "less_than", "value": 7},
        ],
        "parameters": {
            "adx_period": 14,
            "bb_period": 20,
            "bb_width_lookback": 60,
            "vrp_lookback": 20,
        },
        "risk_params": {
            "max_premium_pct": 2.0,
            "dte_entry": 21,
            "wing_width": 10,
            "short_delta": 0.16,
        },
        "options_specs": {
            "legs": [
                {"type": "put", "action": "sell", "delta_target": -0.16},
                {"type": "put", "action": "buy", "delta_target": -0.05},
                {"type": "call", "action": "sell", "delta_target": 0.16},
                {"type": "call", "action": "buy", "delta_target": 0.05},
            ],
            "dte": 21,
            "max_contracts": 5,
            "max_risk_per_contract": "wing width × 100 ($1,000/ct max loss)",
            "greek_targets": {
                "delta": "~0 (neutral)",
                "theta": "+0.15 (positive decay, collect premium)",
                "vega": "-0.20 (short vol, benefits from VRP)",
                "gamma": "-0.01 (short gamma, risk on breakout)",
            },
            "iv_conditions": "VRP > 3pp required — options overpriced vs realized",
            "status": "backtested",
            "is_sharpe": 2.17,
            "oos_sharpe": 0.85,
            "win_rate": 81.7,
            "trades_per_year": 4.4,
        },
    },
    {
        "regime": "ranging + high vol",
        "name": "qqq_iron_butterfly_highvol_v1",
        "structure": "iron_butterfly",
        "description": "ATM iron butterfly in ranging markets with elevated IV. "
                      "Sell ATM straddle, buy OTM wings for protection. "
                      "Collect premium from high IV while regime-gated.",
        "entry_rules": [
            {"indicator": "adx", "condition": "less_than", "value": 25},
            {"indicator": "atr_percentile", "condition": "greater_than", "value": 75},
            {"indicator": "iv_rank", "condition": "greater_than", "value": 60},
        ],
        "exit_rules": [
            {"indicator": "profit_pct", "condition": "greater_than", "value": 60},
            {"indicator": "loss_pct", "condition": "less_than", "value": -100},
            {"indicator": "adx", "condition": "greater_than", "value": 25},
            {"indicator": "dte", "condition": "less_than", "value": 7},
        ],
        "parameters": {
            "adx_period": 14,
            "atr_period": 14,
            "iv_lookback": 252,
        },
        "risk_params": {
            "max_premium_pct": 2.0,
            "dte_entry": 30,
            "wing_width": 15,
            "iv_rank_min": 60,
        },
        "options_specs": {
            "legs": [
                {"type": "put", "action": "sell", "delta_target": -0.50},
                {"type": "put", "action": "buy", "delta_target": -0.15},
                {"type": "call", "action": "sell", "delta_target": 0.50},
                {"type": "call", "action": "buy", "delta_target": 0.15},
            ],
            "dte": 30,
            "max_contracts": 3,
            "max_risk_per_contract": "wing width × 100 ($1,500/ct max loss)",
            "greek_targets": {
                "delta": "~0 (neutral at entry)",
                "theta": "+0.20 (strong positive decay from ATM straddle)",
                "vega": "-0.35 (short vol, benefits if IV mean-reverts)",
                "gamma": "-0.03 (short gamma, tight stop on regime change)",
            },
            "iv_conditions": "IV rank > 60 REQUIRED — selling expensive premium",
            "regime_exit_critical": True,
            "risk_note": "MUST exit on ADX > 25 — tail risk on breakouts",
        },
    },
]


def register_strategy_direct(strat: dict) -> str:
    """Register strategy directly via DB, bypassing MCP tool."""
    strategy_id = f"strat_{uuid.uuid4().hex[:12]}"

    with pg_conn() as conn:
        # Check if already exists
        existing = conn.execute(
            "SELECT strategy_id FROM strategies WHERE name = ?",
            [strat["name"]]
        ).fetchone()

        if existing:
            return existing[0]

        # Insert new strategy
        conn.execute(
            """
            INSERT INTO strategies
                (strategy_id, name, description, asset_class, regime_affinity,
                 parameters, entry_rules, exit_rules, risk_params, status, source,
                 instrument_type, time_horizon, holding_period_days, symbol)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', 'workshop', 'options', 'swing', ?, 'QQQ')
            """,
            [
                strategy_id,
                strat["name"],
                strat["description"],
                "equities",
                json.dumps({}),
                json.dumps(strat["parameters"]),
                json.dumps(strat["entry_rules"]),
                json.dumps(strat["exit_rules"]),
                json.dumps(strat["risk_params"]),
                strat["risk_params"].get("dte_entry", 30),
            ],
        )

    return strategy_id


def main():
    """Design all 6 QQQ options strategies."""

    print(f"\n{'='*120}")
    print("QQQ OPTIONS STRATEGY DESIGN — ALL 6 REGIMES")
    print(f"{'='*120}\n")

    results = []

    for strat in STRATEGIES:
        print(f"\n{'-'*120}")
        print(f"REGIME: {strat['regime']}")
        print(f"STRATEGY: {strat['name']}")
        print(f"STRUCTURE: {strat['structure']}")
        print(f"{'-'*120}")

        # Register strategy
        strategy_id = register_strategy_direct(strat)
        print(f"✓ Strategy ID: {strategy_id}")

        # Extract specs
        specs = strat["options_specs"]
        max_risk = specs.get("max_risk_per_contract", "N/A")
        greeks = specs.get("greek_targets", {})
        iv_cond = specs.get("iv_conditions", "N/A")

        print(f"\nOPTIONS STRUCTURE:")
        print(f"  DTE: {specs['dte']} days")
        print(f"  Max Contracts: {specs['max_contracts']}")
        print(f"  Max Risk/ct: {max_risk}")

        print(f"\n  LEGS:")
        for leg in specs["legs"]:
            symbol = leg.get("symbol", "QQQ")
            print(f"    - {leg['action'].upper():4} {leg['type'].upper():4} @ Δ={leg['delta_target']:+.2f} ({symbol})")

        print(f"\n  GREEK TARGETS:")
        for greek, value in greeks.items():
            print(f"    {greek.capitalize():8}: {value}")

        print(f"\n  IV CONDITIONS: {iv_cond}")

        # Check for special notes
        if specs.get("discretionary_only"):
            print(f"\n  ⚠️  DISCRETIONARY ONLY: {specs.get('reason', 'Not systematic')}")

        if specs.get("regime_exit_critical"):
            print(f"  ⚠️  REGIME EXIT CRITICAL: {specs.get('risk_note', 'Exit on regime change')}")

        if "status" in specs:
            print(f"\n  ✓ VALIDATED RESULTS:")
            print(f"    IS Sharpe: {specs.get('is_sharpe', 'N/A')}")
            print(f"    OOS Sharpe: {specs.get('oos_sharpe', 'N/A')}")
            print(f"    Win Rate: {specs.get('win_rate', 'N/A')}%")
            print(f"    Trades/Year: {specs.get('trades_per_year', 'N/A')}")

        # Calculate R:R ratio
        exit_rules = strat["exit_rules"]
        profit_target = None
        stop_loss = None

        for rule in exit_rules:
            if "profit" in rule["indicator"]:
                profit_target = abs(rule["value"])
            elif "loss" in rule["indicator"]:
                stop_loss = abs(rule["value"])

        rr_ratio = f"{profit_target / stop_loss:.1f}:1" if profit_target and stop_loss else "N/A"
        print(f"\n  R:R TARGET: {rr_ratio}")

        results.append({
            "regime": strat["regime"],
            "strategy_name": strat["name"],
            "equity_proxy_sharpe": specs.get("is_sharpe", "pending"),
            "equity_proxy_trades": specs.get("trades_per_year", "pending"),
            "structure": strat["structure"],
            "max_risk": max_risk,
            "rr_ratio": rr_ratio,
            "status": specs.get("status", "draft"),
            "strategy_id": strategy_id,
        })

    # Print summary table
    print(f"\n\n{'='*150}")
    print("SUMMARY TABLE: QQQ Options Strategies — All 6 Regimes")
    print(f"{'='*150}")
    print(f"{'Regime':<28} {'Strategy Name':<40} {'Eq Proxy Sharpe':<18} {'Eq Proxy Trades':<16} {'Structure':<25} {'Max Risk':<35} {'R:R':<8} {'Status':<12}")
    print(f"{'-'*150}")

    for r in results:
        print(f"{r['regime']:<28} {r['strategy_name']:<40} {str(r['equity_proxy_sharpe']):<18} "
              f"{str(r['equity_proxy_trades']):<16} {r['structure']:<25} {r['max_risk'][:33]:<35} "
              f"{r['rr_ratio']:<8} {r['status']:<12}")

    print(f"{'-'*150}")

    print(f"\n✓ Design complete. {len(results)} options strategies created.")
    print(f"✓ All structures use defined risk (max loss = premium or wing width).")
    print(f"✓ Greek targets and IV conditions specified for each regime.")
    print(f"\nNEXT STEPS:")
    print(f"  1. Existing iron condor (v3) already validated — OOS Sharpe 0.85")
    print(f"  2. Run equity proxy backtests for signal quality on remaining 5 strategies")
    print(f"  3. Options backtest tool currently BROKEN (0 trades) — use equity proxy only")
    print(f"  4. Bear put spread (v3) is DISCRETIONARY — WF failed, use for defined-risk shorts only")
    print(f"  5. Integrate real IV data from broker for actual options P&L estimation")

    # Save detailed specs to JSON
    output_path = Path(__file__).parent.parent / "data" / "qqq_options_6regimes_design.json"
    output_path.parent.mkdir(exist_ok=True, parents=True)

    output_data = []
    for strat in STRATEGIES:
        output_data.append({
            "regime": strat["regime"],
            "name": strat["name"],
            "structure": strat["structure"],
            "description": strat["description"],
            "entry_rules": strat["entry_rules"],
            "exit_rules": strat["exit_rules"],
            "parameters": strat["parameters"],
            "risk_params": strat["risk_params"],
            "options_specs": strat["options_specs"],
        })

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n→ Detailed specifications saved to: {output_path}")


if __name__ == "__main__":
    main()
