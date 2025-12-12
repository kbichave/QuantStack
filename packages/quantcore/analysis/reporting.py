"""
Report generation for WTI trading system.

Generates comprehensive reports with clear separation of:
- In-sample (training) results
- Validation results (used for parameter selection)
- Out-of-sample (test) results - the ONLY reliable metrics
- Quant research metrics (IC, alpha decay, cost-adjusted returns)
"""

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

# Import quant research integration
try:
    from quantcore.research.quant_metrics import (
        run_signal_diagnostics,
        generate_quant_report_section,
        QuantResearchReport,
    )

    QUANT_METRICS_AVAILABLE = True
except ImportError:
    QUANT_METRICS_AVAILABLE = False
    logger.warning("Quant metrics module not available")


def generate_report(
    all_data: Dict[str, pd.DataFrame],
    spread_df: pd.DataFrame,
    backtest_results: Dict[str, float],
    regime_models: Dict[str, object],
    rl_agents: Dict[str, object],
    best_params: Optional[Dict[str, float]] = None,
    strategy_results: Optional[pd.DataFrame] = None,
    rl_metrics: Optional[Dict[str, Dict]] = None,
    mc_results: Optional[Dict[str, Any]] = None,
    tuning_results: Optional[Dict[str, Any]] = None,
    validation_results: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate final summary report with clear in-sample vs out-of-sample distinction.

    Args:
        all_data: Dictionary of all fetched data
        spread_df: DataFrame with spread data
        backtest_results: Final backtest results (should be TEST ONLY)
        regime_models: Trained regime detection models
        rl_agents: Trained RL agents
        best_params: Optimized hyperparameters
        strategy_results: Strategy comparison DataFrame
        rl_metrics: RL training metrics
        mc_results: Monte Carlo simulation results
        tuning_results: Results from hyperparameter tuning with split info
        validation_results: Results from leakage detection and validation checks
    """
    logger.info("=" * 70)
    logger.info("FINAL REPORT")
    logger.info("=" * 70)

    report = []
    report.append("=" * 70)
    report.append("  WTI TRADING SYSTEM - TEST RESULTS")
    report.append("=" * 70)
    report.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Data Integrity Warning
    report.append("DATA INTEGRITY NOTICE")
    report.append("-" * 40)
    report.append("  This report uses proper temporal splits to avoid lookahead bias.")
    report.append("  - Parameters selected on VALIDATION data (not test)")
    report.append("  - Final metrics are from TRUE OUT-OF-SAMPLE test data")
    report.append("  - Monte Carlo runs on holdout data separate from tuning")
    report.append("")

    # Validation Results Section
    if validation_results:
        report.append("VALIDATION & LEAKAGE DETECTION")
        report.append("-" * 40)

        for check_name, check_result in validation_results.items():
            status = "PASS" if check_result.get("passed", False) else "FAIL"
            report.append(f"  {check_name.replace('_', ' ').title()}: {status}")

            for detail in check_result.get("details", [])[:3]:
                report.append(f"    - {detail}")

        report.append("")

    # Data Summary
    report.append("DATA SUMMARY")
    report.append("-" * 40)
    for key, df in all_data.items():
        if isinstance(df, pd.DataFrame):
            report.append(f"  {key}: {len(df)} records")
    report.append("")

    # Temporal Split Info (from tuning results if available)
    if tuning_results and "split_info" in tuning_results:
        split = tuning_results["split_info"]
        report.append("TEMPORAL DATA SPLIT (No Lookahead Bias)")
        report.append("-" * 40)
        report.append(
            f"  Train:      {split['train_start']} to {split['train_end']} ({split['train_bars']:,} bars) - ML Model Fitting"
        )
        report.append(
            f"  Validation: {split['val_start']} to {split['val_end']} ({split['val_bars']:,} bars) - Parameter Selection"
        )
        report.append(
            f"  Test:       {split['test_start']} to {split['test_end']} ({split['test_bars']:,} bars) - Final Eval ONLY"
        )
        report.append("")

    # Current Market State
    if not spread_df.empty:
        latest = spread_df.iloc[-1]
        report.append("CURRENT MARKET STATE")
        report.append("-" * 40)
        report.append(f"  WTI Price: ${latest['wti']:.2f}")
        report.append(f"  Brent Price: ${latest['brent']:.2f}")
        report.append(f"  WTI-Brent Spread: ${latest['spread']:.2f}")
        report.append(f"  Spread Z-Score: {latest['spread_zscore']:.2f}")
        report.append(f"  Current Signal: {latest['signal']}")
        report.append("")

    # News Sentiment
    if "NEWS_SENTIMENT" in all_data and not all_data["NEWS_SENTIMENT"].empty:
        news = all_data["NEWS_SENTIMENT"]
        avg_sentiment = news["overall_sentiment_score"].astype(float).mean()
        report.append("NEWS SENTIMENT")
        report.append("-" * 40)
        report.append(f"  Articles Analyzed: {len(news)}")
        report.append(f"  Average Sentiment: {avg_sentiment:.3f}")
        sentiment_label = (
            "BULLISH"
            if avg_sentiment > 0.1
            else "BEARISH" if avg_sentiment < -0.1 else "NEUTRAL"
        )
        report.append(f"  Overall Outlook: {sentiment_label}")
        report.append("")

    # Model Status
    report.append("REGIME MODELS (trained on pre-test data)")
    report.append("-" * 40)
    report.append(
        f"  HMM (4-state): {'✓ Trained' if 'hmm' in regime_models else '✗ Not trained'}"
    )
    report.append(
        f"  Changepoint: {'✓ Trained' if 'changepoint' in regime_models else '✗ Not trained'}"
    )
    report.append(
        f"  TFT (Transformer): {'✓ Trained' if 'tft' in regime_models else '✗ Not trained'}"
    )
    report.append(
        f"  Combined Detector: {'✓ Trained' if 'combined' in regime_models else '✗ Not trained'}"
    )
    report.append("")

    report.append("RL AGENTS (trained on pre-test data)")
    report.append("-" * 40)

    rl_metrics = rl_metrics or {}

    for agent_name, display_name in [
        ("execution", "Execution RL"),
        ("sizing", "Sizing RL"),
        ("spread", "Spread RL"),
    ]:
        if agent_name in rl_metrics and rl_metrics[agent_name].get("trained"):
            m = rl_metrics[agent_name]
            report.append(f"  {display_name}: ✓ Trained")
            report.append(
                f"      Episodes: {m.get('episodes', 0)}, Final Reward: {m.get('final_reward', 0):.2f}, Avg Reward: {m.get('avg_reward', 0):.2f}"
            )
        elif agent_name in rl_metrics and not rl_metrics[agent_name].get("trained"):
            error = rl_metrics[agent_name].get("error", "Unknown error")
            error_short = error[:40] + "..." if len(str(error)) > 40 else error
            report.append(f"  {display_name}: ✗ Failed ({error_short})")
        else:
            report.append(f"  {display_name}: ✗ Not trained")
    report.append("")

    # Optimized Parameters
    if best_params:
        report.append("OPTIMIZED PARAMETERS (selected on VALIDATION set)")
        report.append("-" * 40)
        report.append("  NOTE: Parameters chosen based on validation performance,")
        report.append("        NOT test performance. This avoids lookahead bias.")
        report.append("")
        for k, v in best_params.items():
            if k == "spread_cost":
                report.append(f"  {k}: ${v:.4f}/barrel (production cost)")
            else:
                report.append(f"  {k}: {v}")
        report.append("")

    # Validation vs Test Comparison (if available)
    if (
        tuning_results
        and "validation_results" in tuning_results
        and "test_results" in tuning_results
    ):
        val_res = tuning_results["validation_results"]
        test_res = tuning_results["test_results"]

        report.append("VALIDATION vs TEST COMPARISON")
        report.append("-" * 40)
        report.append(f"  {'Metric':<20} {'Validation':>15} {'Test (Unseen)':>15}")
        report.append(f"  {'-' * 50}")
        report.append(
            f"  {'Sharpe Ratio':<20} {val_res.get('sharpe_ratio', 0):>15.2f} {test_res.get('sharpe_ratio', 0):>15.2f}"
        )
        report.append(
            f"  {'Return':<20} {val_res.get('total_return_pct', 0):>14.1f}% {test_res.get('total_return_pct', 0):>14.1f}%"
        )
        report.append(
            f"  {'Win Rate':<20} {val_res.get('win_rate', 0):>14.1f}% {test_res.get('win_rate', 0):>14.1f}%"
        )
        report.append(
            f"  {'Max Drawdown':<20} {val_res.get('max_drawdown', 0):>14.1f}% {test_res.get('max_drawdown', 0):>14.1f}%"
        )
        report.append(
            f"  {'Total Trades':<20} {val_res.get('total_trades', 0):>15} {test_res.get('total_trades', 0):>15}"
        )

        # Warn if significant degradation
        val_sharpe = val_res.get("sharpe_ratio", 0)
        test_sharpe = test_res.get("sharpe_ratio", 0)
        if val_sharpe > 0 and test_sharpe < val_sharpe * 0.5:
            report.append("")
            report.append(
                "  ⚠️  WARNING: Significant degradation from validation to test."
            )
            report.append(
                "     This may indicate overfitting to the validation period."
            )
        report.append("")

    # Monte Carlo Results (on holdout)
    if mc_results and "error" not in mc_results:
        holdout_start = mc_results.get("holdout_start", "unknown")
        holdout_bars = mc_results.get("holdout_bars", 0)
        report.append(
            f"MONTE CARLO ROBUSTNESS TEST ({mc_results.get('n_simulations', 500)} simulations)"
        )
        report.append("-" * 40)
        report.append(f"  Holdout Period: {holdout_start}+ ({holdout_bars} bars)")
        report.append(f"  NOTE: Run on data SEPARATE from parameter tuning")
        report.append("")
        report.append(
            f"  Mean Return: {mc_results['mean_return']:.1f}% ± {mc_results['std_return']:.1f}%"
        )
        report.append(f"  Median Return: {mc_results['median_return']:.1f}%")
        report.append(f"  5th Percentile (Worst 5%): {mc_results['percentile_5']:.1f}%")
        report.append(
            f"  95th Percentile (Best 5%): {mc_results['percentile_95']:.1f}%"
        )
        report.append(f"  Absolute Worst Case: {mc_results['min_return']:.1f}%")
        report.append(f"  Probability of Profit: {mc_results['prob_positive']:.1f}%")
        report.append(
            f"  Win Rate Range: {mc_results.get('min_win_rate', 0):.1f}% - {mc_results.get('max_win_rate', mc_results.get('mean_win_rate', 0)):.1f}%"
        )
        report.append("")

    # Final Test Results (the ONLY reliable metric)
    report.append("=" * 70)
    report.append("  FINAL OUT-OF-SAMPLE TEST RESULTS (TRUE UNSEEN DATA)")
    report.append("=" * 70)
    report.append("")
    report.append("  ⚠️  These are the ONLY reliable metrics - based on data")
    report.append("     that was NEVER used for parameter selection or training.")
    report.append("")
    report.append(
        f"  Initial Capital: ${backtest_results.get('initial_capital', 100000):,.2f}"
    )
    report.append(
        f"  Final Capital: ${backtest_results.get('final_capital', 100000):,.2f}"
    )
    report.append(f"  Total P&L: ${backtest_results.get('total_return', 0):,.2f}")
    report.append(f"  Return: {backtest_results.get('total_return_pct', 0):.2f}%")
    report.append(f"  Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}")
    report.append(f"  Max Drawdown: {backtest_results.get('max_drawdown', 0):.2f}%")
    report.append(f"  Win Rate: {backtest_results.get('win_rate', 0):.1f}%")
    report.append(f"  Total Trades: {backtest_results.get('total_trades', 0)}")
    report.append("")

    # Quant Research Metrics Section
    if QUANT_METRICS_AVAILABLE and not spread_df.empty:
        try:
            # Extract signal and returns for quant analysis
            raw_signal = spread_df.get("signal", pd.Series(dtype=object))

            # Convert string signals to numeric: LONG=1, SHORT=-1, NEUTRAL=0
            signal_map = {"LONG": 1, "SHORT": -1, "NEUTRAL": 0, "FLAT": 0}
            if raw_signal.dtype == object or raw_signal.dtype == str:
                signal = raw_signal.map(signal_map).fillna(0).astype(float)
            else:
                signal = raw_signal.astype(float)

            if "spread" in spread_df.columns:
                returns = spread_df["spread"].pct_change()
            elif "wti" in spread_df.columns:
                returns = spread_df["wti"].pct_change()
            else:
                returns = pd.Series(dtype=float)

            if len(signal) > 50 and len(returns) > 50:
                quant_report = run_signal_diagnostics(signal, returns, cost_bps=5.0)

                report.append("QUANT RESEARCH METRICS")
                report.append("=" * 70)
                report.append("")
                report.append("  Signal Quality:")
                report.append(
                    f"    Information Coefficient (IC): {quant_report.ic:.4f}"
                )
                report.append(f"    IC Information Ratio: {quant_report.ic_ir:.2f}")
                report.append(
                    f"    Alpha Half-Life: {quant_report.half_life_bars} bars"
                )
                report.append("")

                # Alpha Decay Curve
                report.append("  Alpha Decay Analysis:")
                if not quant_report.alpha_decay_curve.empty:
                    decay_curve = quant_report.alpha_decay_curve
                    report.append(
                        f"    Lag 1 IC:  {decay_curve.iloc[0]:.4f}"
                        if len(decay_curve) > 0
                        else "    Lag 1 IC:  N/A"
                    )
                    report.append(
                        f"    Lag 5 IC:  {decay_curve.iloc[4]:.4f}"
                        if len(decay_curve) > 4
                        else "    Lag 5 IC:  N/A"
                    )
                    report.append(
                        f"    Lag 10 IC: {decay_curve.iloc[9]:.4f}"
                        if len(decay_curve) > 9
                        else "    Lag 10 IC: N/A"
                    )
                    report.append(
                        f"    Lag 20 IC: {decay_curve.iloc[19]:.4f}"
                        if len(decay_curve) > 19
                        else "    Lag 20 IC: N/A"
                    )
                    report.append(f"    Half-Life: {quant_report.half_life_bars} bars")
                else:
                    report.append("    [Alpha decay curve not available]")
                report.append("")

                report.append("  Performance (Gross vs Net):")
                report.append(f"    Sharpe (Gross): {quant_report.sharpe_ratio:.2f}")
                report.append(
                    f"    Sharpe (Net of 5bps): {quant_report.sharpe_net:.2f}"
                )
                report.append(f"    Sortino (Gross): {quant_report.sortino_ratio:.2f}")
                report.append(f"    Sortino (Net): {quant_report.sortino_net:.2f}")
                report.append(
                    f"    Annual Return (Gross): {quant_report.annual_return * 100:.1f}%"
                )
                report.append(
                    f"    Annual Volatility: {quant_report.annual_volatility * 100:.1f}%"
                )
                report.append(
                    f"    Max Drawdown: {quant_report.max_drawdown * 100:.1f}%"
                )
                report.append("")

                # Cost Breakdown
                report.append("  Transaction Cost Breakdown:")
                report.append(
                    f"    Commission Cost: {quant_report.commission_cost:.2f} bps"
                )
                report.append(f"    Spread Cost: {quant_report.spread_cost:.2f} bps")
                report.append(f"    Market Impact: {quant_report.impact_cost:.2f} bps")
                report.append(
                    f"    Total Round-Trip: {quant_report.total_costs_bps:.1f} bps"
                )
                report.append("")

                report.append("  Turnover Analysis:")
                report.append(
                    f"    Annual Turnover: {quant_report.annual_turnover:.2f}x"
                )
                report.append(
                    f"    Avg Holding Period: {quant_report.avg_holding_days:.1f} days"
                )
                report.append("")

                report.append("  Statistical Tests:")
                report.append(
                    f"    Returns Stationary: {'Yes' if quant_report.is_stationary else 'No'}"
                )
                report.append(f"    ADF p-value: {quant_report.adf_pvalue:.4f}")
                report.append("")
        except Exception as e:
            report.append(f"  [Quant metrics unavailable: {str(e)[:50]}]")
            report.append("")

    # Strategy Comparison
    if strategy_results is not None and not strategy_results.empty:
        report.append("STRATEGY COMPARISON (Out-of-Sample Test Data)")
        report.append("=" * 70)

        if not spread_df.empty:
            valid_data = spread_df.dropna(subset=["spread_zscore"])
            train_data = valid_data[valid_data.index < "2021-01-01"]
            test_data = valid_data[valid_data.index >= "2021-01-01"]

            report.append("")
            report.append("  TEMPORAL DATA SPLIT (No Lookahead Bias)")
            report.append(
                f"  Train:      {len(train_data):,} bars ({train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')}) - Parameters tuned here"
            )
            report.append(
                f"  Test:       {len(test_data):,} bars ({test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}) ← UNSEEN"
            )

        for strategy_type in ["Rule-Based", "ML-Based", "RL-Based", "Benchmark"]:
            type_results = strategy_results[
                strategy_results.get("type", "") == strategy_type
            ]
            if type_results.empty:
                continue

            report.append(f"\n  {strategy_type.upper()}")
            report.append(f"  {'-' * 82}")
            report.append(
                f"  {'Strategy':<23} {'P&L':>12} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7} {'Win%':>7} {'Trades':>7}"
            )

            for _, row in type_results.iterrows():
                pnl = row.get(
                    "total_pnl", row.get("total_return_pct", 0) / 100 * 100000
                )
                report.append(
                    f"  {row['strategy']:<23} ${pnl:>10,.0f} {row['total_return_pct']:>7.1f}% "
                    f"{row['sharpe_ratio']:>7.2f} {row['max_drawdown']:>6.1f}% "
                    f"{row['win_rate']:>6.1f}% {int(row['total_trades']):>7}"
                )

        report.append("")

        # Best per category
        rule_based = strategy_results[strategy_results.get("type", "") == "Rule-Based"]
        ml_based = strategy_results[strategy_results.get("type", "") == "ML-Based"]
        rl_based = strategy_results[strategy_results.get("type", "") == "RL-Based"]

        if not rule_based.empty:
            best_rule = rule_based.loc[rule_based["sharpe_ratio"].idxmax()]
            report.append(
                f"  🥇 Best Rule-Based: {best_rule['strategy']} (Sharpe: {best_rule['sharpe_ratio']:.2f})"
            )

        if not ml_based.empty:
            best_ml = ml_based.loc[ml_based["sharpe_ratio"].idxmax()]
            report.append(
                f"  🥇 Best ML-Based:   {best_ml['strategy']} (Sharpe: {best_ml['sharpe_ratio']:.2f})"
            )

        if not rl_based.empty:
            best_rl = rl_based.loc[rl_based["sharpe_ratio"].idxmax()]
            report.append(
                f"  🥇 Best RL-Based:   {best_rl['strategy']} (Sharpe: {best_rl['sharpe_ratio']:.2f})"
            )

        best = strategy_results.loc[strategy_results["sharpe_ratio"].idxmax()]
        report.append(f"\n  🏆 OVERALL BEST: {best['strategy']}")
        report.append(
            f"     Type: {best.get('type', 'Unknown')}, Sharpe: {best['sharpe_ratio']:.2f}, Return: {best['total_return_pct']:.1f}%"
        )
        report.append("")

    report.append("=" * 70)

    report_text = "\n".join(report)
    print(report_text)

    return report_text
