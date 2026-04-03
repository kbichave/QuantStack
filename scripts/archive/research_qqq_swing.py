#!/usr/bin/env python3
"""
QQQ Equity Swing Research Pipeline — Direct Engine Execution

Bypasses MCP server (quantstack-research not running) and calls the underlying
engines directly: strategy registration via PostgreSQL, backtest engine,
walk-forward validator, and IC computation.

Hypotheses:
  1. QQQ_HVBULL_CORRECTION_BOUNCE (mean-reversion in HIGH_VOL_BULL)
  2. QQQ_BREAKDOWN_SHORT (momentum short on 200-MA breach)
"""

import json
import uuid
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from quantstack.config.timeframes import Timeframe
from quantstack.core.backtesting.engine import BacktestConfig, BacktestEngine
from quantstack.core.features.technical_indicators import TechnicalIndicators
from quantstack.data.storage import DataStore
from quantstack.db import pg_conn
from quantstack.strategies.signal_generator import (
    generate_signals_from_rules,
    fetch_price_data,
)

# ---------------------------------------------------------------------------
# Strategy Definitions
# ---------------------------------------------------------------------------

STRATEGIES = {
    "QQQ_HVBULL_CORRECTION_BOUNCE": {
        "name": "QQQ_HVBULL_CORRECTION_BOUNCE",
        "description": (
            "Mean-reversion long in HIGH_VOL_BULL HMM regime. RSI<35 + below SMA-200 "
            "+ above BB lower band + volume surge marks washout zone where CTA shorts "
            "cover and vol-targeting funds rebalance. 3-7 day hold targeting SMA-200 reclaim."
        ),
        "asset_class": "equities",
        "instrument_type": "equity",
        "time_horizon": "swing",
        "holding_period_days": 5,
        "source": "workshop",
        "regime_affinity": {
            "trending_up_normal": 0.3,
            "trending_up_high": 0.8,
            "trending_down_normal": 0.5,
            "ranging_low": 0.2,
            "ranging_high": 0.7,
        },
        "parameters": {
            "setup_tf": "1d",
            "trigger_tf": "1h",
            "setup_rsi_threshold": 35,
            "trigger_rsi_threshold": 38,
            "sma_proximity_pct": 3.0,
            "stop_loss_atr": 1.5,
            "take_profit_atr": 2.5,
            "time_stop_days": 7,
            "max_trigger_wait_days": 3,
            "economic_mechanism": (
                "In HIGH_VOL_BULL HMM regimes, 8-15% drawdowns structurally attract "
                "vol-targeting fund rebalancing and systematic 401k buying. RSI<35 + BB "
                "lower band marks the washout zone where CTA shorts cover. Counter-party: "
                "momentum longs who panic-sell."
            ),
        },
        "entry_rules": [
            {"indicator": "rsi", "condition": "below", "value": 35, "timeframe": "daily", "type": "prerequisite"},
            {"indicator": "close", "condition": "below", "value": "sma_200", "timeframe": "daily", "type": "prerequisite"},
            {"indicator": "close", "condition": "above", "value": "bb_lower", "timeframe": "daily", "type": "prerequisite"},
        ],
        "exit_rules": [
            {"type": "take_profit", "atr_multiple": 2.5},
            {"type": "stop_loss", "atr_multiple": 1.5},
            {"type": "time_stop", "days": 7},
        ],
        "risk_params": {
            "stop_loss_atr": 1.5,
            "take_profit_atr": 2.5,
            "time_stop_days": 7,
            "position_size_pct": 0.05,
        },
    },
    "QQQ_BREAKDOWN_SHORT": {
        "name": "QQQ_BREAKDOWN_SHORT",
        "description": (
            "Momentum short on confirmed 200-MA breakdown with ADX trend strength. "
            "CTA trend-following short additions triggered by 200-day MA breach. "
            "Sector rotation tech->industrials/energy is structurally driven by AI capex "
            "skepticism. 7-10 day hold."
        ),
        "asset_class": "equities",
        "instrument_type": "equity",
        "time_horizon": "swing",
        "holding_period_days": 7,
        "source": "workshop",
        "regime_affinity": {
            "trending_down_normal": 0.9,
            "trending_down_high": 0.7,
            "trending_up_normal": 0.1,
            "ranging_low": 0.2,
            "ranging_high": 0.3,
        },
        "parameters": {
            "direction": "SHORT",
            "setup_tf": "1d",
            "trigger_tf": "1h",
            "setup_rsi_threshold": 50,
            "trigger_rsi_threshold": 45,
            "sma_proximity_pct": 2.0,
            "stop_loss_atr": 1.5,
            "take_profit_atr": 2.0,
            "time_stop_days": 10,
            "max_trigger_wait_days": 3,
            "economic_mechanism": (
                "200-day MA breach systematically triggers CTA trend-following short "
                "additions (~$50-100B across all CTA programs). Sector rotation out of "
                "tech is structurally driven (AI capex skepticism + margin compression). "
                "Counter-party: passive buy-and-hold rebalancers who add slowly."
            ),
        },
        "entry_rules": [
            {"indicator": "close", "condition": "below", "value": "sma_200", "timeframe": "daily", "type": "prerequisite"},
            {"indicator": "adx", "condition": "above", "value": 25, "timeframe": "daily", "type": "prerequisite"},
            {"indicator": "minus_di", "condition": "above", "value": "plus_di", "timeframe": "daily", "type": "prerequisite"},
        ],
        "exit_rules": [
            {"type": "take_profit", "atr_multiple": 2.0},
            {"type": "stop_loss", "atr_multiple": 1.5},
            {"type": "time_stop", "days": 10},
        ],
        "risk_params": {
            "stop_loss_atr": 1.5,
            "take_profit_atr": 2.0,
            "time_stop_days": 10,
            "position_size_pct": 0.05,
        },
    },
}


def register_strategy(strat_def: dict) -> str:
    """Register strategy directly in PostgreSQL. Returns strategy_id."""
    strategy_id = f"strat_{uuid.uuid4().hex[:12]}"
    name = strat_def["name"]

    with pg_conn() as conn:
        row = conn.execute(
            "SELECT strategy_id FROM strategies WHERE name = %s", [name]
        ).fetchone()
        if row:
            logger.info(f"Strategy '{name}' already exists: {row[0]}")
            return row[0]

        conn.execute(
            """
            INSERT INTO strategies
                (strategy_id, name, description, asset_class, regime_affinity,
                 parameters, entry_rules, exit_rules, risk_params, status, source,
                 instrument_type, time_horizon, holding_period_days)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'draft', %s, %s, %s, %s)
            """,
            [
                strategy_id,
                name,
                strat_def["description"],
                strat_def["asset_class"],
                json.dumps(strat_def.get("regime_affinity", {})),
                json.dumps(strat_def["parameters"]),
                json.dumps(strat_def["entry_rules"]),
                json.dumps(strat_def["exit_rules"]),
                json.dumps(strat_def.get("risk_params", {})),
                strat_def.get("source", "workshop"),
                strat_def.get("instrument_type", "equity"),
                strat_def.get("time_horizon", "swing"),
                strat_def.get("holding_period_days", 5),
            ],
        )
    logger.info(f"Registered strategy {strategy_id}: {name}")
    return strategy_id


def load_price_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load OHLCV data with technical indicators computed."""
    df = fetch_price_data(symbol, start_date, end_date, timeframe="daily")
    if df is None or df.empty:
        raise ValueError(f"No price data for {symbol} [{start_date}..{end_date}]")
    logger.info(f"Loaded {len(df)} bars for {symbol} [{df.index[0]} .. {df.index[-1]}]")
    return df


def run_single_backtest(
    strategy_id: str,
    strat_def: dict,
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100_000.0,
    position_size_pct: float = 0.05,
) -> dict:
    """Run backtest using BacktestEngine directly."""
    price_data = load_price_data(symbol, start_date, end_date)

    entry_rules = strat_def["entry_rules"]
    exit_rules = strat_def["exit_rules"]
    parameters = strat_def["parameters"]

    signals = generate_signals_from_rules(price_data, entry_rules, exit_rules, parameters)

    # Count how many entry signals were generated
    signal_count = 0
    if "signal" in signals.columns:
        signal_count = int((signals["signal"] != 0).sum())
    elif "entry_signal" in signals.columns:
        signal_count = int(signals["entry_signal"].sum())
    logger.info(f"Generated {signal_count} raw entry signals for {symbol}")

    config = BacktestConfig(
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        commission_per_trade=1.0,
        slippage_pct=0.001,
    )
    engine = BacktestEngine(config=config)
    result = engine.run(signals, price_data)

    calmar = 0.0
    if result.max_drawdown > 0:
        calmar = (result.total_return / 100.0) / (result.max_drawdown / 100.0)

    avg_pnl = 0.0
    if result.trades:
        avg_pnl = float(np.mean([t["pnl"] for t in result.trades]))

    summary = {
        "strategy_id": strategy_id,
        "strategy_name": strat_def["name"],
        "symbol": symbol,
        "total_trades": result.total_trades,
        "win_rate": round(result.win_rate, 4),
        "sharpe_ratio": round(result.sharpe_ratio, 4),
        "max_drawdown_pct": round(result.max_drawdown, 2),
        "total_return_pct": round(result.total_return, 2),
        "profit_factor": round(result.profit_factor, 4),
        "calmar_ratio": round(calmar, 4),
        "avg_trade_pnl": round(avg_pnl, 2),
        "raw_signals": signal_count,
        "start_date": str(price_data.index[0].date()) if hasattr(price_data.index[0], "date") else str(price_data.index[0]),
        "end_date": str(price_data.index[-1].date()) if hasattr(price_data.index[-1], "date") else str(price_data.index[-1]),
        "bars_tested": len(price_data),
    }

    # Persist to DB
    with pg_conn() as conn:
        conn.execute(
            "UPDATE strategies SET backtest_summary = %s, status = CASE WHEN status = 'draft' THEN 'backtested' ELSE status END, updated_at = CURRENT_TIMESTAMP WHERE strategy_id = %s",
            [json.dumps(summary), strategy_id],
        )

    return summary


def run_walk_forward(
    strategy_id: str,
    strat_def: dict,
    symbol: str,
    start_date: str,
    end_date: str,
    n_splits: int = 3,
    initial_capital: float = 100_000.0,
    position_size_pct: float = 0.05,
) -> dict:
    """Run walk-forward validation with purged splits."""
    price_data = load_price_data(symbol, start_date, end_date)
    entry_rules = strat_def["entry_rules"]
    exit_rules = strat_def["exit_rules"]
    parameters = strat_def["parameters"]

    n_bars = len(price_data)
    # Minimum: 252 bars per test fold, 504 bars minimum train
    test_size = max(126, n_bars // (n_splits + 2))
    min_train = max(252, test_size * 2)
    embargo = max(5, int(n_bars * 0.01))

    logger.info(
        f"Walk-forward: {n_splits} folds, test_size={test_size}, "
        f"min_train={min_train}, embargo={embargo}, total_bars={n_bars}"
    )

    train_sharpes = []
    test_sharpes = []
    train_returns = []
    test_returns = []
    fold_details = []

    for fold_idx in range(n_splits):
        test_end_idx = n_bars - (n_splits - 1 - fold_idx) * test_size
        test_start_idx = test_end_idx - test_size
        train_end_idx = test_start_idx - embargo

        if fold_idx == 0:
            train_start_idx = 0
        else:
            train_start_idx = 0  # expanding window

        if train_end_idx - train_start_idx < min_train:
            logger.warning(
                f"Fold {fold_idx}: train too small "
                f"({train_end_idx - train_start_idx} < {min_train}), skipping"
            )
            continue

        train_data = price_data.iloc[train_start_idx:train_end_idx]
        test_data = price_data.iloc[test_start_idx:test_end_idx]

        config = BacktestConfig(
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            commission_per_trade=1.0,
            slippage_pct=0.001,
        )

        # Train fold
        train_signals = generate_signals_from_rules(train_data, entry_rules, exit_rules, parameters)
        train_engine = BacktestEngine(config=config)
        train_result = train_engine.run(train_signals, train_data)

        # Test fold (OOS)
        test_signals = generate_signals_from_rules(test_data, entry_rules, exit_rules, parameters)
        test_engine = BacktestEngine(config=config)
        test_result = test_engine.run(test_signals, test_data)

        fold_info = {
            "fold": fold_idx,
            "train_period": f"{train_data.index[0].date()} .. {train_data.index[-1].date()}",
            "test_period": f"{test_data.index[0].date()} .. {test_data.index[-1].date()}",
            "train_bars": len(train_data),
            "test_bars": len(test_data),
            "train_trades": train_result.total_trades,
            "test_trades": test_result.total_trades,
            "train_sharpe": round(train_result.sharpe_ratio, 4),
            "test_sharpe": round(test_result.sharpe_ratio, 4),
            "train_return_pct": round(train_result.total_return, 2),
            "test_return_pct": round(test_result.total_return, 2),
            "train_win_rate": round(train_result.win_rate, 4),
            "test_win_rate": round(test_result.win_rate, 4),
            "train_pf": round(train_result.profit_factor, 4),
            "test_pf": round(test_result.profit_factor, 4),
        }
        fold_details.append(fold_info)
        train_sharpes.append(train_result.sharpe_ratio)
        test_sharpes.append(test_result.sharpe_ratio)
        train_returns.append(train_result.total_return)
        test_returns.append(test_result.total_return)

    if not fold_details:
        return {
            "success": False,
            "error": "No valid walk-forward folds (insufficient data)",
            "strategy_id": strategy_id,
        }

    avg_train_sharpe = float(np.mean(train_sharpes))
    avg_test_sharpe = float(np.mean(test_sharpes))
    overfit_ratio = avg_train_sharpe / avg_test_sharpe if avg_test_sharpe != 0 else float("inf")

    # Probability of backtest overfitting (simplified):
    # PBO = fraction of folds where test Sharpe < 0
    pbo = sum(1 for s in test_sharpes if s < 0) / len(test_sharpes) if test_sharpes else 1.0

    wf_summary = {
        "strategy_id": strategy_id,
        "strategy_name": strat_def["name"],
        "symbol": symbol,
        "n_folds": len(fold_details),
        "avg_train_sharpe": round(avg_train_sharpe, 4),
        "avg_test_sharpe": round(avg_test_sharpe, 4),
        "overfit_ratio": round(overfit_ratio, 4),
        "pbo": round(pbo, 4),
        "avg_train_return_pct": round(float(np.mean(train_returns)), 2),
        "avg_test_return_pct": round(float(np.mean(test_returns)), 2),
        "folds": fold_details,
        "pass_criteria": {
            "oos_sharpe_gt_0.3": avg_test_sharpe > 0.3,
            "overfit_ratio_lt_2.0": overfit_ratio < 2.0,
            "pbo_lt_0.40": pbo < 0.40,
        },
        "passed": avg_test_sharpe > 0.3 and overfit_ratio < 2.0 and pbo < 0.40,
    }

    return wf_summary


def compute_ic(
    strat_def: dict,
    symbol: str,
    start_date: str,
    end_date: str,
    forward_days: int = 5,
) -> dict:
    """Compute Information Coefficient for the primary signal vs forward returns."""
    price_data = load_price_data(symbol, start_date, end_date)

    entry_rules = strat_def["entry_rules"]
    exit_rules = strat_def["exit_rules"]
    parameters = strat_def["parameters"]

    signals = generate_signals_from_rules(price_data, entry_rules, exit_rules, parameters)

    # Build a composite signal score (number of entry conditions met)
    # For IC we need a continuous signal, not just binary
    # Use the signal column if available
    signal_col = None
    for col in ["signal", "entry_signal", "composite_score"]:
        if col in signals.columns:
            signal_col = col
            break

    if signal_col is None:
        return {
            "success": False,
            "error": "No signal column found in generated signals",
            "strategy_name": strat_def["name"],
        }

    signal_series = signals[signal_col].astype(float)

    # Compute forward returns
    fwd_returns = price_data["close"].pct_change(forward_days).shift(-forward_days)

    # Align and drop NaN
    combined = pd.DataFrame({
        "signal": signal_series,
        "fwd_return": fwd_returns,
    }).dropna()

    if len(combined) < 50:
        return {
            "success": False,
            "error": f"Too few observations for IC: {len(combined)}",
            "strategy_name": strat_def["name"],
        }

    # Rank IC (Spearman)
    from scipy.stats import spearmanr
    ic_val, ic_pval = spearmanr(combined["signal"], combined["fwd_return"])

    # Also compute on signal-active days only
    active = combined[combined["signal"] != 0]
    ic_active = None
    if len(active) >= 10:
        ic_a, pval_a = spearmanr(active["signal"], active["fwd_return"])
        ic_active = {"ic": round(ic_a, 6), "pval": round(pval_a, 6), "n": len(active)}

    # Alpha decay: IC at multiple horizons
    decay = {}
    for h in [1, 3, 5, 7, 10, 15, 20]:
        fwd_h = price_data["close"].pct_change(h).shift(-h)
        combo = pd.DataFrame({"signal": signal_series, "fwd": fwd_h}).dropna()
        if len(combo) >= 50:
            ic_h, _ = spearmanr(combo["signal"], combo["fwd"])
            decay[f"{h}d"] = round(ic_h, 6)

    return {
        "success": True,
        "strategy_name": strat_def["name"],
        "symbol": symbol,
        "forward_days": forward_days,
        "rank_ic": round(ic_val, 6),
        "ic_pval": round(ic_pval, 6),
        "n_observations": len(combined),
        "ic_active_days": ic_active,
        "alpha_decay": decay,
        "signal_col_used": signal_col,
        "pass_criteria": {
            "ic_gt_0.02": abs(ic_val) > 0.02,
        },
    }


def main():
    symbol = "QQQ"
    is_start = "2022-01-01"
    is_end = "2026-01-01"

    results = {}

    for strat_name, strat_def in STRATEGIES.items():
        print(f"\n{'='*80}")
        print(f"STRATEGY: {strat_name}")
        print(f"{'='*80}")

        # Step 1: Register
        print(f"\n--- Step 1: Register ---")
        try:
            strategy_id = register_strategy(strat_def)
            print(f"  strategy_id = {strategy_id}")
        except Exception as e:
            print(f"  REGISTRATION FAILED: {e}")
            traceback.print_exc()
            results[strat_name] = {"error": f"Registration failed: {e}"}
            continue

        # Step 2: Backtest (IS)
        print(f"\n--- Step 2: Backtest (IS) ---")
        try:
            bt = run_single_backtest(
                strategy_id, strat_def, symbol, is_start, is_end,
            )
            print(f"  Sharpe:       {bt['sharpe_ratio']}")
            print(f"  Total trades: {bt['total_trades']}")
            print(f"  Win rate:     {bt['win_rate']}")
            print(f"  Return:       {bt['total_return_pct']}%")
            print(f"  Max DD:       {bt['max_drawdown_pct']}%")
            print(f"  Profit Factor:{bt['profit_factor']}")
            print(f"  Calmar:       {bt['calmar_ratio']}")
            print(f"  Raw signals:  {bt['raw_signals']}")
        except Exception as e:
            print(f"  BACKTEST FAILED: {e}")
            traceback.print_exc()
            bt = {"error": str(e)}

        # Step 3: IC
        print(f"\n--- Step 3: Information Coefficient ---")
        try:
            ic = compute_ic(strat_def, symbol, is_start, is_end, forward_days=5)
            if ic.get("success"):
                print(f"  Rank IC:      {ic['rank_ic']}")
                print(f"  IC p-value:   {ic['ic_pval']}")
                print(f"  N obs:        {ic['n_observations']}")
                print(f"  Alpha decay:  {ic.get('alpha_decay', {})}")
                if ic.get("ic_active_days"):
                    print(f"  IC (active):  {ic['ic_active_days']}")
            else:
                print(f"  IC FAILED: {ic.get('error')}")
        except Exception as e:
            print(f"  IC FAILED: {e}")
            traceback.print_exc()
            ic = {"error": str(e)}

        # Step 4: Walk-forward (OOS)
        print(f"\n--- Step 4: Walk-Forward Validation ---")
        try:
            wf = run_walk_forward(
                strategy_id, strat_def, symbol, is_start, is_end, n_splits=3,
            )
            if "error" not in wf:
                print(f"  Avg train Sharpe: {wf['avg_train_sharpe']}")
                print(f"  Avg OOS Sharpe:   {wf['avg_test_sharpe']}")
                print(f"  Overfit ratio:    {wf['overfit_ratio']}")
                print(f"  PBO:              {wf['pbo']}")
                print(f"  PASSED:           {wf['passed']}")
                for fold in wf["folds"]:
                    print(f"    Fold {fold['fold']}: train={fold['train_sharpe']:.4f} / "
                          f"test={fold['test_sharpe']:.4f} | "
                          f"trades: {fold['train_trades']}/{fold['test_trades']} | "
                          f"{fold['test_period']}")
            else:
                print(f"  WF FAILED: {wf['error']}")
        except Exception as e:
            print(f"  WF FAILED: {e}")
            traceback.print_exc()
            wf = {"error": str(e)}

        results[strat_name] = {
            "strategy_id": strategy_id,
            "backtest": bt,
            "ic": ic,
            "walkforward": wf,
        }

    # Final summary
    print(f"\n{'='*80}")
    print("RESEARCH SUMMARY")
    print(f"{'='*80}")
    for strat_name, res in results.items():
        print(f"\n{strat_name}:")
        if "error" in res:
            print(f"  ERROR: {res['error']}")
            continue
        sid = res.get("strategy_id", "?")
        bt = res.get("backtest", {})
        ic = res.get("ic", {})
        wf = res.get("walkforward", {})
        print(f"  strategy_id:    {sid}")
        print(f"  IS Sharpe:      {bt.get('sharpe_ratio', 'N/A')}")
        print(f"  IS trades:      {bt.get('total_trades', 'N/A')}")
        print(f"  Rank IC:        {ic.get('rank_ic', 'N/A')}")
        print(f"  OOS Sharpe:     {wf.get('avg_test_sharpe', 'N/A')}")
        print(f"  Overfit ratio:  {wf.get('overfit_ratio', 'N/A')}")
        print(f"  PBO:            {wf.get('pbo', 'N/A')}")
        passed_wf = wf.get("passed", False)
        print(f"  WF passed:      {passed_wf}")

        # Gate assessment
        bt_pass = bt.get("sharpe_ratio", 0) > 0.5 and bt.get("total_trades", 0) > 20 and bt.get("profit_factor", 0) > 1.2
        ic_pass = ic.get("success") and abs(ic.get("rank_ic", 0)) > 0.02
        print(f"  Gate 1 (IC):    {'PASS' if ic_pass else 'FAIL'}")
        print(f"  Gate 2 (IS):    {'PASS' if bt_pass else 'FAIL'}")
        print(f"  Gate 3 (OOS):   {'PASS' if passed_wf else 'FAIL'}")

    return results


if __name__ == "__main__":
    results = main()
