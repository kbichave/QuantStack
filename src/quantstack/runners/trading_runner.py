"""Trading graph continuous runner — async cycle every 5 min (market hours)."""

import asyncio
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from quantstack.health.heartbeat import write_heartbeat
from quantstack.observability.instrumentation import langfuse_trace_context
from quantstack.runners import get_cycle_interval

logger = logging.getLogger(__name__)

WATCHDOG_TIMEOUT = 600  # seconds


def save_checkpoint(graph_name: str, cycle_number: int, duration: float,
                    status: str, error_message: str | None = None) -> None:
    """Write cycle checkpoint to PostgreSQL. Best-effort — never crashes the loop."""
    try:
        from quantstack.db import db_conn
        with db_conn() as conn:
            conn.execute(
                """INSERT INTO graph_checkpoints
                   (graph_name, cycle_number, duration_seconds, status, error_message, created_at)
                   VALUES (%s, %s, %s, %s, %s, NOW())
                   ON CONFLICT (graph_name, cycle_number) DO UPDATE
                   SET duration_seconds = EXCLUDED.duration_seconds,
                       status = EXCLUDED.status,
                       error_message = EXCLUDED.error_message,
                       created_at = NOW()""",
                [graph_name, cycle_number, duration, status, error_message],
            )
    except Exception:
        logger.exception("Failed to save checkpoint for %s cycle %d", graph_name, cycle_number)


async def run_loop(
    graph_builder: Callable,
    initial_state_builder: Callable,
    shutdown,
    graph_name: str = "trading",
    watchdog_timeout: int = WATCHDOG_TIMEOUT,
) -> None:
    """Async runner loop. Rebuilds graph each cycle, invokes with timeout.

    Args:
        graph_builder: Callable returning a compiled StateGraph.
        initial_state_builder: Callable returning initial state dict for the graph.
        shutdown: GracefulShutdown instance — loop checks shutdown.should_stop.
        graph_name: Name used for heartbeat, checkpoint, and interval lookup.
        watchdog_timeout: Max seconds per graph invocation before timeout.
    """
    cycle_number = 0
    consecutive_failures = 0

    while not shutdown.should_stop:
        interval = get_cycle_interval(graph_name)

        if interval is None:
            logger.info("%s runner paused (weekend/holiday) — polling in 60s", graph_name)
            await asyncio.sleep(60)
            continue

        cycle_number += 1
        start = time.monotonic()
        status = "success"
        error_msg = None

        try:
            graph = graph_builder()
            initial_state = initial_state_builder(cycle_number)

            date_str = datetime.now().strftime("%Y-%m-%d")
            thread_id = f"{graph_name}-{date_str}-cycle-{cycle_number}"

            with langfuse_trace_context(
                session_id=f"{graph_name}-{date_str}",
                tags=[graph_name, f"cycle-{cycle_number}"],
                name=f"{graph_name}_cycle",
            ) as trace:
                final_state = await asyncio.wait_for(
                    graph.ainvoke(
                        initial_state,
                        config={"configurable": {"thread_id": thread_id}},
                    ),
                    timeout=watchdog_timeout,
                )

            # Inspect errors from the graph execution
            errors = final_state.get("errors", [])
            if errors:
                logger.warning(
                    "%s cycle %d completed with %d errors: %s",
                    graph_name, cycle_number, len(errors), errors[:3],
                )

            await asyncio.to_thread(write_heartbeat, graph_name)
            consecutive_failures = 0

        except asyncio.TimeoutError:
            status = "timeout"
            error_msg = f"Cycle timed out after {watchdog_timeout}s"
            consecutive_failures += 1
            logger.critical(
                "%s cycle %d timed out after %ds",
                graph_name, cycle_number, watchdog_timeout,
            )

        except Exception:
            status = "error"
            error_msg = traceback.format_exc()
            consecutive_failures += 1
            logger.exception(
                "%s cycle %d failed (consecutive: %d)",
                graph_name, cycle_number, consecutive_failures,
            )

        if consecutive_failures >= 3:
            logger.critical(
                "%s has failed %d consecutive cycles — supervisor should investigate",
                graph_name, consecutive_failures,
            )

        elapsed = time.monotonic() - start
        await asyncio.to_thread(
            save_checkpoint, graph_name, cycle_number, elapsed, status, error_msg,
        )

        sleep_time = max(0, interval - elapsed)
        if sleep_time > 0 and not shutdown.should_stop:
            await asyncio.sleep(sleep_time)


async def async_main() -> None:
    """Async entry point for the trading runner."""
    from quantstack.health.shutdown import GracefulShutdown
    from quantstack.observability.instrumentation import setup_instrumentation

    try:
        setup_instrumentation()
    except Exception:
        logger.warning("Langfuse instrumentation unavailable — continuing without tracing")

    # Run DB migrations (idempotent, creates graph_checkpoints etc.)
    try:
        from quantstack.db import db_conn, run_migrations
        with db_conn() as conn:
            run_migrations(conn)
    except Exception:
        logger.warning("DB migrations failed — checkpointing will be degraded")

    shutdown = GracefulShutdown()
    shutdown.install_async(asyncio.get_running_loop())

    from langgraph.checkpoint.memory import MemorySaver
    from quantstack.graphs.config_watcher import ConfigWatcher
    from quantstack.graphs.trading import build_trading_graph

    yaml_path = Path(__file__).resolve().parent.parent / "graphs" / "trading" / "config" / "agents.yaml"
    config_watcher = ConfigWatcher(yaml_path)
    checkpointer = MemorySaver()

    def graph_builder():
        return build_trading_graph(config_watcher, checkpointer)

    def initial_state_builder(cycle_number: int) -> dict[str, Any]:
        return {
            "cycle_number": cycle_number,
            "regime": "unknown",
            "portfolio_context": {},
            "errors": [],
            "decisions": [],
        }

    logger.info("Starting trading runner")
    try:
        await run_loop(graph_builder, initial_state_builder, shutdown, graph_name="trading")
    finally:
        config_watcher.stop()
    logger.info("Trading runner stopped")


def main() -> None:
    """Entry point: python -m quantstack.runners.trading_runner"""
    asyncio.run(async_main())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
