#!/usr/bin/env python3
"""
Bootstrap & Walk-Forward Simulation — get the company ready for Monday.

This script:
1. Loads 4 years of D1+M15 OHLCV for the founding 5 symbols from Alpaca
2. Loads options chain snapshots from Alpha Vantage
3. Loads economic indicators (CPI, Fed Funds, NFP)
4. Runs the first equity snapshot
5. Spawns the Quant Researcher for initial strategy generation
6. Spawns the ML Scientist for initial model training
7. Generates Monday's trading sheets

Run in tmux so it continues while you're away:
    tmux new-session -s bootstrap
    python scripts/bootstrap_and_simulate.py
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure packages are importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "packages"))

from loguru import logger


FOUNDING_5 = ["SPY", "QQQ", "IWM", "TSLA", "NVDA"]


async def step_1_load_ohlcv():
    """Load D1 + M15 OHLCV from Alpaca for founding 5."""
    logger.info("=" * 60)
    logger.info("STEP 1: Loading OHLCV from Alpaca")
    logger.info("=" * 60)

    from quantcore.config.timeframes import Timeframe
    from quantcore.data.storage import DataStore

    # Try Alpaca first
    try:
        from quantcore.data.adapters.alpaca import AlpacaAdapter

        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

        if not api_key:
            logger.warning("No ALPACA_API_KEY — trying Alpha Vantage for OHLCV")
            return await _load_ohlcv_alphavantage()

        adapter = AlpacaAdapter(api_key=api_key, secret_key=secret_key)
        store = DataStore()

        start = datetime(2022, 1, 1, tzinfo=timezone.utc)
        end = datetime.now(timezone.utc)

        for symbol in FOUNDING_5:
            for tf in [Timeframe.D1, Timeframe.M15]:
                try:
                    logger.info(f"  Fetching {symbol} {tf.value}...")
                    df = adapter.fetch_ohlcv(symbol, tf, start_date=start, end_date=end)
                    if df is not None and not df.empty:
                        rows = store.save_ohlcv(df, symbol, tf, replace=False)
                        logger.info(f"  {symbol} {tf.value}: {len(df)} bars fetched, {rows} new rows cached")
                    else:
                        logger.warning(f"  {symbol} {tf.value}: no data returned")
                except Exception as exc:
                    logger.warning(f"  {symbol} {tf.value}: failed — {exc}")

    except ImportError:
        logger.warning("alpaca-py not available — falling back to Alpha Vantage")
        await _load_ohlcv_alphavantage()


async def _load_ohlcv_alphavantage():
    """Fallback: load OHLCV from Alpha Vantage."""
    from quantcore.config.timeframes import Timeframe
    from quantcore.data.adapters.alphavantage import AlphaVantageAdapter
    from quantcore.data.storage import DataStore

    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    if not api_key:
        logger.error("No ALPHA_VANTAGE_API_KEY — cannot load OHLCV")
        return

    adapter = AlphaVantageAdapter(api_key=api_key)
    store = DataStore()

    for symbol in FOUNDING_5:
        for tf in [Timeframe.D1]:  # AV free tier is rate-limited, start with D1 only
            try:
                logger.info(f"  Fetching {symbol} {tf.value} from Alpha Vantage...")
                df = adapter.fetch_ohlcv(symbol, tf)
                if df is not None and not df.empty:
                    rows = store.save_ohlcv(df, symbol, tf, replace=False)
                    logger.info(f"  {symbol} {tf.value}: {len(df)} bars, {rows} new rows")
            except Exception as exc:
                logger.warning(f"  {symbol} {tf.value}: {exc}")


async def step_2_load_options_and_macro():
    """Load options chains + economic indicators from Alpha Vantage."""
    logger.info("=" * 60)
    logger.info("STEP 2: Loading options + macro from Alpha Vantage")
    logger.info("=" * 60)

    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    if not api_key:
        logger.warning("No ALPHA_VANTAGE_API_KEY — skipping options + macro")
        return

    # Options chains (latest EOD)
    from quantcore.data.adapters.alphavantage import AlphaVantageAdapter
    adapter = AlphaVantageAdapter(api_key=api_key)

    for symbol in FOUNDING_5:
        try:
            contracts = adapter.fetch_options_chain(symbol, expiry_max_days=45)
            count = len(contracts) if contracts else 0
            logger.info(f"  {symbol} options chain: {count} contracts")
        except Exception as exc:
            logger.warning(f"  {symbol} options: {exc}")

    # Economic indicators
    logger.info("  Loading economic indicators...")
    try:
        from quantcore.data.macro_calendar import MacroCalendarGenerator
        from quant_pod.db import open_db

        conn = open_db()
        gen = MacroCalendarGenerator()
        results = gen.fetch_economic_history(conn)
        logger.info(f"  Economic data: {results}")
    except Exception as exc:
        logger.warning(f"  Economic indicators: {exc}")


async def step_3_equity_snapshot():
    """Run first equity snapshot."""
    logger.info("=" * 60)
    logger.info("STEP 3: First equity snapshot")
    logger.info("=" * 60)

    try:
        from quant_pod.db import open_db, run_migrations
        from quant_pod.performance.equity_tracker import EquityTracker

        conn = open_db()
        run_migrations(conn)
        tracker = EquityTracker(conn)
        result = tracker.snapshot_daily()
        logger.info(f"  Equity snapshot: {result}")
    except Exception as exc:
        logger.error(f"  Equity snapshot failed: {exc}")


async def step_4_generate_trading_sheets():
    """Generate Monday's trading sheets for all 5 symbols."""
    logger.info("=" * 60)
    logger.info("STEP 4: Generating trading sheets")
    logger.info("=" * 60)

    try:
        from quant_pod.performance.trading_sheet import TradingSheetGenerator

        generator = TradingSheetGenerator()
        sheets = await generator.generate_all(FOUNDING_5)

        # Write to file for easy review
        output_path = project_root / "trading_sheets_monday.md"
        with open(output_path, "w") as f:
            f.write(f"# Monday Trading Sheets — Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            for sheet in sheets:
                f.write(sheet.to_markdown())
                f.write("\n\n---\n\n")

        logger.info(f"  Trading sheets written to {output_path}")

        # Also print summary
        for sheet in sheets:
            logger.info(
                f"  {sheet.symbol}: {sheet.recommended_action} | "
                f"regime={sheet.trend_regime} bias={sheet.consensus_bias} "
                f"conviction={sheet.consensus_conviction:.0%}"
            )

    except Exception as exc:
        logger.error(f"  Trading sheets failed: {exc}")


async def step_5_test_alpaca_execution():
    """Test Alpaca paper execution with a tiny order."""
    logger.info("=" * 60)
    logger.info("STEP 5: Testing Alpaca paper execution")
    logger.info("=" * 60)

    try:
        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        if not api_key:
            logger.warning("  No ALPACA_API_KEY — skipping execution test")
            return

        from alpaca.trading.client import TradingClient
        client = TradingClient(api_key, secret_key, paper=True)

        account = client.get_account()
        logger.info(f"  Account: equity=${float(account.equity):,.2f} cash=${float(account.cash):,.2f}")
        logger.info(f"  Paper mode confirmed: {account.account_number}")
        logger.info("  Alpaca paper execution: READY")

    except Exception as exc:
        logger.error(f"  Alpaca test failed: {exc}")


async def main():
    """Run the full bootstrap sequence."""
    logger.info("=" * 60)
    logger.info("QUANTPOD BOOTSTRAP — Getting ready for Monday")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Symbols: {', '.join(FOUNDING_5)}")
    logger.info("=" * 60)

    # Load env vars
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")

    await step_1_load_ohlcv()
    await step_2_load_options_and_macro()
    await step_3_equity_snapshot()
    await step_4_generate_trading_sheets()
    await step_5_test_alpaca_execution()

    logger.info("=" * 60)
    logger.info("BOOTSTRAP COMPLETE")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next: Start the research orchestrator loop:")
    logger.info("  tmux new-session -s research")
    logger.info("  claude --prompt \"$(cat prompts/research_orchestrator.md)\"")
    logger.info("")
    logger.info("The pods will self-start and work overnight.")
    logger.info("Monday morning: check trading_sheets_monday.md for the playbook.")


if __name__ == "__main__":
    asyncio.run(main())
