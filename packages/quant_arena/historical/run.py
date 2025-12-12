# Copyright 2024 QuantArena Contributors
# SPDX-License-Identifier: Apache-2.0

"""
CLI entrypoint for historical QuantArena simulation.

All agents use OpenAI gpt-4o for reasoning and trading decisions.
API key loaded from OPENAI_API_KEY in .env file.

Usage:
    python -m quant_arena.historical.run
    python -m quant_arena.historical.run --symbols SPY,QQQ --equity 100000
    python -m quant_arena.historical.run --start 2010-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load .env file for OPENAI_API_KEY
_project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(_project_root / ".env")

from quant_arena.historical.config import HistoricalConfig
from quant_arena.historical.engine import HistoricalEngine


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    logger.remove()

    level = "DEBUG" if verbose else "INFO"
    format_str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(sys.stderr, format=format_str, level=level, colorize=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Historical QuantArena - Multi-agent trading simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with defaults (SPY, QQQ, IWM, WTI, BRENT)
    python -m quant_arena.historical.run
    
    # Custom symbols and equity
    python -m quant_arena.historical.run --symbols SPY,QQQ --equity 50000
    
    # Specific date range
    python -m quant_arena.historical.run --start 2015-01-01 --end 2023-12-31
    
    # Save results to JSON
    python -m quant_arena.historical.run --output results.json
        """,
    )

    # Symbol configuration
    parser.add_argument(
        "--symbols",
        type=str,
        default="SPY,QQQ,IWM,WTI,BRENT",
        help="Comma-separated list of symbols (default: SPY,QQQ,IWM,WTI,BRENT)",
    )

    # Date range
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (default: earliest available)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: today)",
    )

    # Capital settings
    parser.add_argument(
        "--equity",
        type=float,
        default=100_000,
        help="Initial equity in USD (default: 100000)",
    )
    parser.add_argument(
        "--max-position",
        type=float,
        default=0.20,
        help="Max position size as fraction (default: 0.20)",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.15,
        help="Max drawdown before halt (default: 0.15)",
    )

    # Transaction costs
    parser.add_argument(
        "--slippage",
        type=float,
        default=5.0,
        help="Slippage in basis points (default: 5)",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.005,
        help="Commission per share (default: 0.005)",
    )

    # Policy settings
    parser.add_argument(
        "--policy-update",
        type=str,
        choices=["monthly", "quarterly", "never"],
        default="monthly",
        help="Policy update frequency (default: monthly)",
    )
    parser.add_argument(
        "--pods",
        type=str,
        default="trend_following,mean_reversion,momentum,breakout,volatility",
        help="Active strategy pods (default: all 5 strategies)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON (optional)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to DuckDB experience store (optional)",
    )

    # Learning system
    parser.add_argument(
        "--enable-learning",
        action="store_true",
        default=True,
        help="Enable learning system (MetaOrchestrator, lesson injection)",
    )
    parser.add_argument(
        "--no-learning",
        action="store_true",
        help="Disable learning system",
    )

    # Multi-timeframe settings
    parser.add_argument(
        "--enable-mtf",
        action="store_true",
        default=True,
        help="Enable multi-timeframe analysis (Weekly->Daily->4H->1H cascade)",
    )
    parser.add_argument(
        "--no-mtf",
        action="store_true",
        help="Disable multi-timeframe analysis (daily-only mode)",
    )
    parser.add_argument(
        "--exec-timeframe",
        type=str,
        choices=["daily", "4h", "1h"],
        default="4h",
        help="Execution timeframe (default: 4h)",
    )
    parser.add_argument(
        "--use-super-trader",
        action="store_true",
        default=True,
        help="Enable SuperTrader as final decision aggregator",
    )
    parser.add_argument(
        "--no-super-trader",
        action="store_true",
        help="Disable SuperTrader (use individual pod decisions)",
    )

    # Speed optimization
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use gpt-4o-mini for faster/cheaper execution (recommended for long simulations)",
    )
    # NOTE: --no-llm removed - LLM reasoning is required for all trading decisions
    # All agents must use LLM for intelligent analysis

    # Verbosity
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> HistoricalConfig:
    """Build configuration from arguments."""
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    pods = [p.strip() for p in args.pods.split(",")]

    # Determine learning flag (--no-learning overrides --enable-learning)
    enable_learning = args.enable_learning and not args.no_learning

    # Determine MTF flags
    enable_mtf = args.enable_mtf and not args.no_mtf
    use_super_trader = args.use_super_trader and not args.no_super_trader

    # LLM reasoning is always required - no rule-based fallbacks
    use_llm = True

    return HistoricalConfig(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        initial_equity=args.equity,
        max_position_pct=args.max_position,
        max_drawdown_halt_pct=args.max_drawdown,
        slippage_bps=args.slippage,
        commission_per_share=args.commission,
        policy_update_frequency=args.policy_update,
        active_pods=pods,
        db_path=args.db_path,
        enable_learning=enable_learning,
        enable_mtf=enable_mtf,
        execution_timeframe=args.exec_timeframe,
        use_super_trader=use_super_trader if use_llm else False,
        use_fast_llm=getattr(args, "fast", False),
        use_llm=use_llm,
    )


def print_banner() -> None:
    """Print startup banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     █████╗ ██╗     ██████╗ ██╗  ██╗ █████╗                    ║
║    ██╔══██╗██║     ██╔══██╗██║  ██║██╔══██╗                   ║
║    ███████║██║     ██████╔╝███████║███████║                   ║
║    ██╔══██║██║     ██╔═══╝ ██╔══██║██╔══██║                   ║
║    ██║  ██║███████╗██║     ██║  ██║██║  ██║                   ║
║    ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝                   ║
║                                                               ║
║    █████╗ ██████╗ ███████╗███╗   ██╗ █████╗                   ║
║   ██╔══██╗██╔══██╗██╔════╝████╗  ██║██╔══██╗                  ║
║   ███████║██████╔╝█████╗  ██╔██╗ ██║███████║                  ║
║   ██╔══██║██╔══██╗██╔══╝  ██║╚██╗██║██╔══██║                  ║
║   ██║  ██║██║  ██║███████╗██║ ╚████║██║  ██║                  ║
║   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝                  ║
║                                                               ║
║       LLM-Powered Multi-Agent Trading Simulation              ║
║              (OpenAI gpt-4o Reasoning Agents)                 ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

    # Check for API key
    if os.getenv("OPENAI_API_KEY"):
        print("🤖 OpenAI API key detected - using LLM-powered agents\n")
    else:
        print("⚠️  Warning: OPENAI_API_KEY not found in .env file")
        print("   Agents will fail without a valid API key!\n")


def print_config(config: HistoricalConfig) -> None:
    """Print configuration summary."""
    print("\n📋 Configuration:")
    print(f"   Symbols:        {', '.join(config.symbols)}")
    print(
        f"   Date range:     {config.start_date or 'earliest'} to {config.end_date or 'today'}"
    )
    print(f"   Initial equity: ${config.initial_equity:,.0f}")
    print(f"   Max position:   {config.max_position_pct:.0%}")
    print(f"   Max drawdown:   {config.max_drawdown_halt_pct:.0%}")
    print(f"   Slippage:       {config.slippage_bps} bps")
    print(f"   Commission:     ${config.commission_per_share}/share")
    print(f"   Policy update:  {config.policy_update_frequency}")
    print(f"   Active pods:    {', '.join(config.active_pods)}")

    # Learning settings
    learning_status = "🧠 ENABLED" if config.enable_learning else "❌ DISABLED"
    print(f"   Learning:       {learning_status}")

    # MTF settings
    mtf_status = "📊 ENABLED" if getattr(config, "enable_mtf", False) else "❌ DISABLED"
    exec_tf = getattr(config, "execution_timeframe", "daily")
    supertrader = "✅ YES" if getattr(config, "use_super_trader", False) else "❌ NO"
    print(f"   MTF Analysis:   {mtf_status}")
    print(f"   Exec Timeframe: {exec_tf.upper()}")
    print(f"   SuperTrader:    {supertrader}")
    print()


def print_result(result) -> None:
    """Print simulation results."""
    print("\n" + "=" * 60)
    print("📊 SIMULATION RESULTS")
    print("=" * 60)

    print(f"\n📅 Period: {result.start_date} to {result.end_date}")
    print(f"   Trading days: {result.trading_days:,}")
    print(f"   Symbols: {', '.join(result.symbols)}")

    print(f"\n💰 Performance:")
    print(f"   Initial equity:  ${result.initial_equity:>12,.0f}")
    print(f"   Final equity:    ${result.final_equity:>12,.0f}")
    print(f"   Total return:    {result.total_return:>12.1%}")

    if result.sharpe_ratio is not None:
        print(f"   Sharpe ratio:    {result.sharpe_ratio:>12.2f}")

    print(f"\n⚠️  Risk:")
    print(f"   Max drawdown:    {result.max_drawdown:>12.1%}")

    print(f"\n📈 Trading:")
    print(f"   Total trades:    {result.total_trades:>12,}")
    print(f"   Win rate:        {result.win_rate:>12.1%}")

    print("\n" + "=" * 60)


def save_results(result, output_path: str) -> None:
    """Save results to JSON file."""
    output = {
        "start_date": result.start_date.isoformat(),
        "end_date": result.end_date.isoformat(),
        "initial_equity": result.initial_equity,
        "final_equity": result.final_equity,
        "total_return": result.total_return,
        "max_drawdown": result.max_drawdown,
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "sharpe_ratio": result.sharpe_ratio,
        "trading_days": result.trading_days,
        "symbols": result.symbols,
    }

    path = Path(output_path)
    path.write_text(json.dumps(output, indent=2))
    print(f"\n💾 Results saved to: {path}")


async def run_simulation(config: HistoricalConfig) -> None:
    """Run the simulation."""
    engine = HistoricalEngine(config)

    # Progress callback
    last_progress = [0]

    def on_progress(current_date, portfolio_state, day_result):
        progress = engine.progress
        if progress - last_progress[0] >= 0.10:  # Log every 10%
            print(
                f"   Progress: {progress:.0%} | "
                f"Date: {current_date} | "
                f"Equity: ${portfolio_state.equity:,.0f}"
            )
            last_progress[0] = progress

    engine.set_callbacks(on_day_complete=on_progress)

    print("\n🚀 Starting simulation...\n")

    result = await engine.run()

    return result


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    print_banner()

    config = build_config(args)
    print_config(config)

    try:
        result = asyncio.run(run_simulation(config))
        print_result(result)

        if args.output:
            save_results(result, args.output)

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"Simulation failed: {e}")
        return 1


def run_historical_quant_arena() -> int:
    """Entry point for module execution."""
    return main()


if __name__ == "__main__":
    sys.exit(main())
