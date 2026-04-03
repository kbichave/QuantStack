"""Research graph continuous runner — async cycle every 10 min (market hours)."""

import asyncio
import logging
from pathlib import Path
from typing import Any

from quantstack.runners.trading_runner import run_loop

logger = logging.getLogger(__name__)

WATCHDOG_TIMEOUT = 900  # seconds — research cycles are longer


async def async_main() -> None:
    """Async entry point for the research runner."""
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
    from quantstack.graphs.research import build_research_graph

    yaml_path = Path(__file__).resolve().parent.parent / "graphs" / "research" / "config" / "agents.yaml"
    config_watcher = ConfigWatcher(yaml_path)
    checkpointer = MemorySaver()

    def graph_builder():
        return build_research_graph(config_watcher, checkpointer)

    def initial_state_builder(cycle_number: int) -> dict[str, Any]:
        return {
            "cycle_number": cycle_number,
            "regime": "unknown",
            "errors": [],
            "decisions": [],
        }

    logger.info("Starting research runner")
    try:
        await run_loop(
            graph_builder, initial_state_builder, shutdown,
            graph_name="research", watchdog_timeout=WATCHDOG_TIMEOUT,
        )
    finally:
        config_watcher.stop()
    logger.info("Research runner stopped")


def main() -> None:
    """Entry point: python -m quantstack.runners.research_runner"""
    asyncio.run(async_main())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
