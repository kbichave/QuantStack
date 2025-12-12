"""
Report generation for equity pipeline.

Provides dataclasses for results and text report generation.
Includes quant research metrics integration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
from loguru import logger

# Import quant research integration
try:
    from quantcore.research.quant_metrics import (
        run_signal_diagnostics,
        QuantResearchReport,
    )

    QUANT_METRICS_AVAILABLE = True
except ImportError:
    QUANT_METRICS_AVAILABLE = False


@dataclass
class TickerStrategyResult:
    """Result for a single ticker-strategy combination."""

    ticker: str
    strategy: str
    pnl: float
    num_trades: int
    win_rate: float
    sharpe: float


@dataclass
class StrategyResult:
    """Aggregated results for a strategy across all tickers."""

    strategy_name: str
    strategy_type: str  # "rule-based" or "ml"
    total_pnl: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    avg_trade_pnl: float
    per_ticker: Dict[str, TickerStrategyResult] = field(default_factory=dict)
    best_ticker: str = ""
    worst_ticker: str = ""
    train_metrics: Dict[str, Any] = field(default_factory=dict)
    # Quant research metrics
    ic: float = 0.0
    ic_ir: float = 0.0
    alpha_halflife: int = 0
    sortino_ratio: float = 0.0
    annual_turnover: float = 0.0


@dataclass
class PipelineReport:
    """Complete pipeline report."""

    timestamp: str
    symbols_count: int
    data_summary: Dict[str, str]
    split_info: Dict[str, str]
    strategy_results: Dict[str, StrategyResult]
    ticker_breakdown: Dict[str, Dict[str, float]]  # ticker -> strategy -> pnl
    best_strategy: str
    recommendations: List[str]


def generate_text_report(report: PipelineReport, output_path: Path) -> None:
    """
    Generate detailed text report with per-ticker breakdown.

    Args:
        report: PipelineReport with all results
        output_path: Path to save the report
    """
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("  EQUITY SIGNAL PIPELINE - FULL RESULTS")
    lines.append("=" * 80)
    lines.append(f"  Generated: {report.timestamp}")
    lines.append(f"  Symbols: {report.symbols_count}")
    lines.append(f"  Trade Size: 100 shares, $0 commission")
    lines.append("")

    # Data Integrity Notice
    lines.append("DATA INTEGRITY NOTICE")
    lines.append("-" * 40)
    lines.append("  - Proper 60/20/20 temporal train/val/test split")
    lines.append("  - No lookahead bias")
    lines.append("  - All metrics from TRUE OUT-OF-SAMPLE test data")
    lines.append("")

    # Data Summary
    lines.append("DATA SUMMARY")
    lines.append("-" * 40)
    for key, value in report.data_summary.items():
        lines.append(f"  {key}: {value}")
    lines.append("")

    # Split Info
    lines.append("TEMPORAL SPLIT")
    lines.append("-" * 40)
    for key, value in report.split_info.items():
        lines.append(f"  {key}: {value}")
    lines.append("")

    # Strategy Summary
    lines.append("=" * 80)
    lines.append("  STRATEGY COMPARISON (Out-of-Sample)")
    lines.append("=" * 80)
    lines.append("")

    header = f"  {'Strategy':<20} {'P&L':>12} {'Return':>8} {'MaxDD':>8} {'Win%':>7} {'Trades':>7} {'Best':>8} {'Worst':>8}"
    lines.append(header)
    lines.append("  " + "-" * 78)

    for name, result in sorted(
        report.strategy_results.items(), key=lambda x: -x[1].total_pnl
    ):
        best_pnl = (
            result.per_ticker.get(
                result.best_ticker, TickerStrategyResult("", "", 0, 0, 0, 0)
            ).pnl
            if result.best_ticker
            else 0
        )
        worst_pnl = (
            result.per_ticker.get(
                result.worst_ticker, TickerStrategyResult("", "", 0, 0, 0, 0)
            ).pnl
            if result.worst_ticker
            else 0
        )
        lines.append(
            f"  {name:<20} ${result.total_pnl:>10,.0f} {result.total_return:>7.1%} "
            f"{result.max_drawdown:>7.1%} {result.win_rate:>6.1%} {result.num_trades:>7} "
            f"{result.best_ticker:>8} {result.worst_ticker:>8}"
        )

    lines.append("")
    lines.append(f"  BEST STRATEGY: {report.best_strategy}")
    lines.append("")

    # Quant Research Metrics (if available)
    if QUANT_METRICS_AVAILABLE:
        has_quant_metrics = any(r.ic != 0 for r in report.strategy_results.values())
        if has_quant_metrics:
            lines.append("=" * 80)
            lines.append("  QUANT RESEARCH METRICS")
            lines.append("=" * 80)
            lines.append("")
            header = f"  {'Strategy':<20} {'IC':>8} {'IC IR':>8} {'Sortino':>8} {'HalfLife':>8} {'Turnover':>10}"
            lines.append(header)
            lines.append("  " + "-" * 64)

            for name, result in sorted(
                report.strategy_results.items(), key=lambda x: -x[1].total_pnl
            ):
                lines.append(
                    f"  {name:<20} {result.ic:>8.4f} {result.ic_ir:>8.2f} "
                    f"{result.sortino_ratio:>8.2f} {result.alpha_halflife:>8} {result.annual_turnover:>9.2f}x"
                )
            lines.append("")

    # Per-Ticker Breakdown
    lines.append("=" * 80)
    lines.append("  RESULTS BY TICKER")
    lines.append("=" * 80)
    lines.append("")

    for ticker in sorted(report.ticker_breakdown.keys()):
        ticker_results = report.ticker_breakdown[ticker]
        lines.append(f"  {ticker}:")
        for strategy, pnl in sorted(ticker_results.items(), key=lambda x: -x[1]):
            lines.append(f"    {strategy:<20} PnL=${pnl:>10,.0f}")
        lines.append("")

    # Recommendations
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 40)
    for rec in report.recommendations:
        lines.append(f"  - {rec}")
    lines.append("")

    lines.append("=" * 80)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Report saved to: {output_path}")

    # Print to console
    print("\n".join(lines))


def enrich_with_quant_metrics(
    result: StrategyResult,
    signals: pd.Series,
    returns: pd.Series,
    cost_bps: float = 5.0,
) -> StrategyResult:
    """
    Enrich strategy result with quant research metrics.

    Args:
        result: StrategyResult to enrich
        signals: Position signals
        returns: Asset returns
        cost_bps: Transaction cost assumption

    Returns:
        Enriched StrategyResult
    """
    if not QUANT_METRICS_AVAILABLE:
        return result

    try:
        quant_report = run_signal_diagnostics(signals, returns, cost_bps)
        result.ic = quant_report.ic
        result.ic_ir = quant_report.ic_ir
        result.alpha_halflife = quant_report.half_life_bars
        result.sortino_ratio = quant_report.sortino_ratio
        result.annual_turnover = quant_report.annual_turnover
    except Exception as e:
        logger.warning(
            f"Failed to compute quant metrics for {result.strategy_name}: {e}"
        )

    return result
