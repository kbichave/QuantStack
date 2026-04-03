"""ML and feature computation tools for LangGraph agents."""

import json

from langchain_core.tools import tool

from quantstack.tools.mcp_bridge._bridge import get_bridge


@tool
async def train_model(
    symbol: str,
    model_type: str = "lightgbm",
    target: str = "forward_return_5d",
) -> str:
    """Train an ML model for signal prediction.

    Args:
        symbol: Ticker symbol.
        model_type: Model type (e.g., "lightgbm", "xgboost", "random_forest").
        target: Target variable to predict.

    Returns JSON with model metrics (Sharpe, IC, accuracy) and model ID.
    """
    bridge = get_bridge()
    result = await bridge.call_quantcore(
        "train_ml_model",
        symbol=symbol,
        model_type=model_type,
        target=target,
    )
    return json.dumps(result, default=str)


@tool
async def compute_features(symbol: str, timeframe: str = "daily") -> str:
    """Compute all available features (200+) for a symbol.

    Includes trend, momentum, volatility, volume, and pattern features.
    Returns JSON with feature values and metadata.
    """
    bridge = get_bridge()
    result = await bridge.call_quantcore(
        "compute_all_features",
        symbol=symbol,
        timeframe=timeframe,
    )
    return json.dumps(result, default=str)
