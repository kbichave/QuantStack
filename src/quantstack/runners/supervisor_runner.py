"""Supervisor graph continuous runner — health checks every 5 min, always on."""

import asyncio
import logging
from pathlib import Path
from typing import Any

from quantstack.runners.trading_runner import run_loop

logger = logging.getLogger(__name__)

WATCHDOG_TIMEOUT = 300  # seconds — supervisor cycles are short


async def async_main() -> None:
    """Async entry point for the supervisor runner."""
    from quantstack.health.shutdown import GracefulShutdown
    from quantstack.observability.instrumentation import setup_instrumentation

    try:
        setup_instrumentation()
    except Exception:
        logger.warning("Langfuse instrumentation unavailable — continuing without tracing")

    shutdown = GracefulShutdown()
    shutdown.install_async(asyncio.get_running_loop())

    from langgraph.checkpoint.memory import MemorySaver
    from quantstack.graphs.config_watcher import ConfigWatcher
    from quantstack.graphs.supervisor import build_supervisor_graph

    yaml_path = Path(__file__).resolve().parent.parent / "graphs" / "supervisor" / "config" / "agents.yaml"
    config_watcher = ConfigWatcher(yaml_path)
    checkpointer = MemorySaver()

    def graph_builder():
        return build_supervisor_graph(config_watcher, checkpointer)

    def initial_state_builder(cycle_number: int) -> dict[str, Any]:
        return {
            "cycle_number": cycle_number,
            "errors": [],
        }

    logger.info("Starting supervisor runner")
    try:
        await run_loop(
            graph_builder, initial_state_builder, shutdown,
            graph_name="supervisor", watchdog_timeout=WATCHDOG_TIMEOUT,
        )
    finally:
        config_watcher.stop()
    logger.info("Supervisor runner stopped")


def main() -> None:
    """Entry point: python -m quantstack.runners.supervisor_runner"""
    asyncio.run(async_main())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
