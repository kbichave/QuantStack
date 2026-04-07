"""Per-agent cost tracking and budget enforcement.

Provides:
- TokenBudgetTracker: in-agent token accumulation with budget enforcement
- compute_cost_usd: estimate USD cost from token counts and model name
- detect_cost_anomaly: flag agents exceeding N-x baseline spend
"""

import logging

logger = logging.getLogger(__name__)

# Model pricing per million tokens (input, output) in USD
# Source: provider pricing pages as of 2026-04
# NOTE: This is intentionally a hardcoded lookup table for cost estimation,
# not a model selection mechanism. Keys must match actual model names.
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-4-5": (0.80, 4.0),
    # OpenAI
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    # Groq (free tier, but track for comparison)
    "llama-3.3-70b-versatile": (0.59, 0.79),
    # Google
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.0-flash": (0.10, 0.40),
}

# Default pricing when model not recognized
_DEFAULT_PRICING = (3.0, 15.0)  # Sonnet-tier as conservative estimate


class TokenBudgetTracker:
    """Track token usage per agent invocation and enforce budget limits.

    Used in agent_executor to accumulate prompt + completion tokens across
    tool-calling rounds and halt the agent if max_tokens is exceeded.
    """

    def __init__(self, max_tokens: int | None):
        self._max_tokens = max_tokens
        self._input_tokens = 0
        self._output_tokens = 0

    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens

    @property
    def input_tokens(self) -> int:
        return self._input_tokens

    @property
    def output_tokens(self) -> int:
        return self._output_tokens

    @property
    def total_tokens(self) -> int:
        return self._input_tokens + self._output_tokens

    @property
    def budget_exceeded(self) -> bool:
        if self._max_tokens is None:
            return False
        return self.total_tokens > self._max_tokens


def _extract_model_name(model: str) -> str:
    """Extract the base model name from a provider/model string."""
    if "/" in model:
        _, model_id = model.split("/", 1)
    else:
        model_id = model
    # Strip version suffixes like -20251001-v1:0
    for known in _MODEL_PRICING:
        if known in model_id:
            return known
    return model_id


def compute_cost_usd(
    input_tokens: int,
    output_tokens: int,
    model: str,
) -> float:
    """Estimate USD cost from token counts and model identifier.

    Uses per-model pricing tables. Falls back to Sonnet-tier pricing
    for unknown models (conservative estimate).
    """
    if input_tokens == 0 and output_tokens == 0:
        return 0.0

    base_name = _extract_model_name(model)
    input_price, output_price = _MODEL_PRICING.get(base_name, _DEFAULT_PRICING)

    cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return cost


def detect_cost_anomaly(
    current_cost: float,
    baseline_avg: float,
    threshold: float = 3.0,
) -> bool:
    """Return True if current_cost exceeds threshold * baseline_avg.

    A zero baseline means no history — any non-zero cost is anomalous.
    """
    if baseline_avg <= 0:
        return current_cost > 0
    return current_cost > (baseline_avg * threshold)
