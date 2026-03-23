# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Bootstrap flow — first-run setup to get the system ready for trading.

Steps:
  1. Load OHLCV (Alpaca primary, Alpha Vantage fallback)
  2. Load options chains + macro indicators from Alpha Vantage
  3. Run first equity snapshot + DB migrations
  4. Generate trading sheets for all symbols
  5. Test broker connectivity (Alpaca paper)

Usage:
    quantpod-bootstrap                          # all steps, default symbols
    quantpod-bootstrap --symbols SPY QQQ AAPL   # custom symbols
    quantpod-bootstrap --skip-broker-test       # skip Alpaca connectivity check
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from quantstack.config.timeframes import Timeframe
from quantstack.data.adapters.alphavantage import AlphaVantageAdapter
from quantstack.data.adapters.alpaca import AlpacaAdapter
from quantstack.data.macro_calendar import MacroCalendarGenerator
from quantstack.data.storage import DataStore
from quantstack.db import open_db, run_migrations
from quantstack.performance.equity_tracker import EquityTracker
from quantstack.performance.trading_sheet import TradingSheetGenerator

from alpaca.trading.client import TradingClient as _AlpacaTradingClient
from dotenv import load_dotenv as _load_dotenv

DEFAULT_SYMBOLS = ["SPY", "QQQ", "IWM", "TSLA", "NVDA"]


class BootstrapFlow:
    """One-time system bootstrap — data loading, equity snapshot, trading sheets."""

    def __init__(
        self,
        symbols: list[str] | None = None,
        skip_broker_test: bool = False,
    ) -> None:
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.skip_broker_test = skip_broker_test
        self.results: dict[str, Any] = {}

    async def run(self) -> dict[str, Any]:
        """Run the full bootstrap sequence. Returns step results."""
        logger.info("=" * 60)
        logger.info("QUANTPOD BOOTSTRAP")
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info("=" * 60)

        await self._load_ohlcv()
        await self._load_options_and_macro()
        await self._equity_snapshot()
        await self._generate_trading_sheets()
        if not self.skip_broker_test:
            await self._test_broker()

        logger.info("=" * 60)
        logger.info("BOOTSTRAP COMPLETE")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Next: Start the autonomous loops:")
        logger.info("  FORCE_LOOPS=1 ./scripts/start_loops.sh all")

        return self.results

    # ------------------------------------------------------------------
    # Step 1: OHLCV
    # ------------------------------------------------------------------

    async def _load_ohlcv(self) -> None:
        logger.info("STEP 1: Loading OHLCV")

        store = DataStore()

        # Try Alpaca first
        api_key = os.environ.get("ALPACA_API_KEY", "")
        if api_key:
            secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
            adapter = AlpacaAdapter(api_key=api_key, secret_key=secret_key)
            start = datetime(2022, 1, 1, tzinfo=timezone.utc)
            end = datetime.now(timezone.utc)

            for symbol in self.symbols:
                for tf in [Timeframe.D1, Timeframe.M15]:
                    try:
                        df = adapter.fetch_ohlcv(symbol, tf, start_date=start, end_date=end)
                        if df is not None and not df.empty:
                            rows = store.save_ohlcv(df, symbol, tf, replace=False)
                            logger.info(f"  {symbol} {tf.value}: {len(df)} bars, {rows} new")
                    except Exception as exc:
                        logger.warning(f"  {symbol} {tf.value}: {exc}")

            self.results["ohlcv"] = "alpaca"
            return

        # Fallback: Alpha Vantage
        await self._load_ohlcv_alphavantage(store)

    async def _load_ohlcv_alphavantage(self, store: Any) -> None:
        api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if not api_key:
            logger.error("No ALPHA_VANTAGE_API_KEY — cannot load OHLCV")
            self.results["ohlcv"] = "skipped"
            return

        adapter = AlphaVantageAdapter(api_key=api_key)
        for symbol in self.symbols:
            try:
                df = adapter.fetch_ohlcv(symbol, Timeframe.D1)
                if df is not None and not df.empty:
                    rows = store.save_ohlcv(df, symbol, Timeframe.D1, replace=False)
                    logger.info(f"  {symbol} D1: {len(df)} bars, {rows} new")
            except Exception as exc:
                logger.warning(f"  {symbol} D1: {exc}")

        self.results["ohlcv"] = "alphavantage"

    # ------------------------------------------------------------------
    # Step 2: Options + Macro
    # ------------------------------------------------------------------

    async def _load_options_and_macro(self) -> None:
        logger.info("STEP 2: Loading options + macro")

        api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if not api_key:
            logger.warning("No ALPHA_VANTAGE_API_KEY — skipping")
            self.results["options_macro"] = "skipped"
            return

        adapter = AlphaVantageAdapter(api_key=api_key)

        for symbol in self.symbols:
            try:
                contracts = adapter.fetch_options_chain(symbol, expiry_max_days=45)
                logger.info(f"  {symbol} options: {len(contracts) if contracts else 0} contracts")
            except Exception as exc:
                logger.warning(f"  {symbol} options: {exc}")

        try:
            conn = open_db()
            results = MacroCalendarGenerator().fetch_economic_history(conn)
            logger.info(f"  Macro: {results}")
            conn.close()
        except Exception as exc:
            logger.warning(f"  Macro: {exc}")

        self.results["options_macro"] = "done"

    # ------------------------------------------------------------------
    # Step 3: Equity Snapshot
    # ------------------------------------------------------------------

    async def _equity_snapshot(self) -> None:
        logger.info("STEP 3: Equity snapshot")

        try:
            conn = open_db()
            run_migrations(conn)
            result = EquityTracker(conn).snapshot_daily()
            logger.info(f"  Snapshot: {result}")
            conn.close()
            self.results["equity_snapshot"] = "done"
        except Exception as exc:
            logger.error(f"  Equity snapshot failed: {exc}")
            self.results["equity_snapshot"] = f"failed: {exc}"

    # ------------------------------------------------------------------
    # Step 4: Trading Sheets
    # ------------------------------------------------------------------

    async def _generate_trading_sheets(self) -> None:
        logger.info("STEP 4: Trading sheets")

        try:
            generator = TradingSheetGenerator()
            sheets = await generator.generate_all(self.symbols)

            output_path = Path("trading_sheets_monday.md")
            with open(output_path, "w") as f:
                f.write(f"# Trading Sheets — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
                for sheet in sheets:
                    f.write(sheet.to_markdown())
                    f.write("\n\n---\n\n")

            logger.info(f"  Written to {output_path}")
            self.results["trading_sheets"] = "done"
        except Exception as exc:
            logger.error(f"  Trading sheets failed: {exc}")
            self.results["trading_sheets"] = f"failed: {exc}"

    # ------------------------------------------------------------------
    # Step 5: Broker Test
    # ------------------------------------------------------------------

    async def _test_broker(self) -> None:
        logger.info("STEP 5: Broker connectivity")

        try:
            api_key = os.environ.get("ALPACA_API_KEY", "")
            if not api_key:
                logger.warning("  No ALPACA_API_KEY — skipping")
                self.results["broker_test"] = "skipped"
                return

            secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
            client = _AlpacaTradingClient(api_key, secret_key, paper=True)
            account = client.get_account()
            logger.info(f"  Equity=${float(account.equity):,.2f} Cash=${float(account.cash):,.2f}")
            self.results["broker_test"] = "ready"
        except Exception as exc:
            logger.error(f"  Broker test failed: {exc}")
            self.results["broker_test"] = f"failed: {exc}"


def main() -> None:
    """CLI entry point for quantpod-bootstrap."""
    parser = argparse.ArgumentParser(description="QuantPod bootstrap — first-run setup")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="Symbols to bootstrap")
    parser.add_argument("--skip-broker-test", action="store_true", help="Skip Alpaca connectivity check")
    args = parser.parse_args()

    _load_dotenv()

    flow = BootstrapFlow(symbols=args.symbols, skip_broker_test=args.skip_broker_test)
    asyncio.run(flow.run())


if __name__ == "__main__":
    main()
