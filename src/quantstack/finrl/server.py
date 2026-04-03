"""
FinRL worker server — exposes RL training/inference via SSE transport.

Runs as a standalone Docker service (finrl-worker) to isolate torch/CUDA
dependencies from the main graph services.

Usage:
    python -m quantstack.finrl.server
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from quantstack.finrl.config import get_finrl_config
from quantstack.finrl.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

_registry: ModelRegistry | None = None


def _get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "finrl-worker", "ts": datetime.now(timezone.utc).isoformat()})


async def sse_stream(request: Request) -> StreamingResponse:
    """SSE endpoint for health probes and event streaming."""

    async def event_generator():
        yield f"data: {json.dumps({'type': 'connected', 'service': 'finrl-worker'})}\n\n"
        while True:
            await asyncio.sleep(30)
            yield f"data: {json.dumps({'type': 'heartbeat', 'ts': datetime.now(timezone.utc).isoformat()})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def list_models(request: Request) -> JSONResponse:
    registry = _get_registry()
    models = registry.list_models()
    return JSONResponse({"models": models})


async def model_status(request: Request) -> JSONResponse:
    model_id = request.path_params["model_id"]
    registry = _get_registry()
    status = registry.get_status(model_id)
    if status is None:
        return JSONResponse({"error": f"Model {model_id} not found"}, status_code=404)
    return JSONResponse(status)


async def config_endpoint(request: Request) -> JSONResponse:
    cfg = get_finrl_config()
    return JSONResponse({
        "config_version": cfg.config_version,
        "shadow_mode_enabled": cfg.shadow_mode_enabled,
        "default_algorithm": cfg.default_algorithm,
        "ensemble_algorithms": cfg.ensemble_algorithms,
    })


routes = [
    Route("/health", health),
    Route("/sse", sse_stream),
    Route("/models", list_models),
    Route("/models/{model_id}", model_status),
    Route("/config", config_endpoint),
]

app = Starlette(routes=routes)


def main():
    import uvicorn

    port = int(os.environ.get("FINRL_PORT", "8090"))
    logger.info("Starting FinRL worker on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
