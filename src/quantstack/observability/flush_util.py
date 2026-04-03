"""Shutdown flush utility for Langfuse traces."""

import logging

logger = logging.getLogger(__name__)


def flush_traces() -> None:
    """Flush all pending Langfuse traces and shut down the client.

    Must be called in the runner's graceful shutdown handler
    before process exit. No-op if instrumentation was never initialized.
    """
    from quantstack.observability.tracing import shutdown
    try:
        shutdown()
        logger.info("Langfuse traces flushed and client shut down")
    except Exception:
        logger.debug("Failed to flush Langfuse traces", exc_info=True)
