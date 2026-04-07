# Section 7: LLM Circuit Breaker & Runtime Failover

## Problem

LLM provider availability is checked only at startup in `provider.py`. If a provider returns a 429 (rate limit), 500 (server error), or times out mid-session, the entire graph cycle crashes with no recovery. Since all three graphs (Trading, Research, Supervisor) depend on LLM calls at every node, a single provider outage halts the entire system.

The existing `get_model_with_fallback()` in `provider.py` handles startup-time validation (checking env vars), but does nothing for runtime errors. The `FALLBACK_ORDER` list (`["bedrock", "anthropic", "openai", "groq", "ollama", "bedrock_groq"]`) already defines provider priority but is only used during initial model resolution, not during invocation failures.

## Design

Create a `CircuitBreaker` class that wraps LLM invocations with per-provider health tracking and automatic failover. Health state is in-memory only (no DB needed) because provider health is transient and resets on restart.

### Failover Logic

1. On retryable error (HTTP 429, 500, timeout): retry the same provider with exponential backoff (2 attempts, delays of 1s then 2s).
2. On 3rd consecutive failure for a provider: mark it as "cooling down" and switch to the next provider in `FALLBACK_ORDER`.
3. Cooled-down providers re-enter rotation after a 5-minute cooldown period.
4. Non-retryable errors (HTTP 400, 401, 403): fail immediately without retry. These indicate bad requests or credential issues that retrying will not fix.

### Per-Provider Health State

Each provider tracks:
- `consecutive_failures`: int, reset to 0 on success
- `cooldown_until`: optional timestamp, set to `now + 5 minutes` when failures reach threshold
- `last_error`: string, for diagnostics

A provider is "available" when `consecutive_failures < 3` and (`cooldown_until` is None or `now > cooldown_until`). When a cooled-down provider's cooldown expires, its `consecutive_failures` resets to 0, giving it a fresh start.

### Integration with LangChain

Use LangChain's `with_fallbacks()` method to chain providers. The CircuitBreaker wraps this chain, intercepting errors to update health state and skip cooled-down providers. The existing `get_chat_model()` function in `provider.py` is the integration point: it returns a chat model that the CircuitBreaker wraps with the fallback chain.

## Dependencies

- **Phase 1 complete** (sections 01-06): This section assumes the baseline safety hardening is in place.
- **`FALLBACK_ORDER` in `provider.py`**: Already defined. The circuit breaker reads this list to determine failover order.
- **`_instantiate_chat_model()` in `provider.py`**: Used to create chat model instances for each fallback provider.
- **No dependency on Section 8 (email alerting)**: Circuit breaker logs warnings/errors. Email integration for CRITICAL alerts is added in Section 15 (layered circuit breaker), not here.

## Tests First

All tests go in a single file. They should be fast (no real LLM calls), using mocks for provider invocations.

```python
# tests/llm/test_circuit_breaker.py

"""Tests for LLM circuit breaker and runtime failover.

All tests mock LLM provider calls — no real API requests.
"""

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ---- Retryable error handling ----

# Test: retryable error (429) retries same provider with backoff
#   Simulate a 429 on first call, success on second.
#   Assert the provider was called twice (not immediately failed over).
#   Assert backoff delay was applied (mock time.sleep or asyncio.sleep).

# Test: retryable error (500) retries same provider with backoff
#   Same structure as 429 test but with 500 status code.
#   Verifies both common retryable codes are handled identically.

# ---- Failover after exhausting retries ----

# Test: 3rd consecutive failure switches to next provider in FALLBACK_ORDER
#   Simulate 3 consecutive failures on provider A.
#   Assert provider B (next in FALLBACK_ORDER) is used for the 4th call.
#   Assert provider A is marked as cooling down.

# Test: failed provider enters 5-minute cooldown
#   Trigger 3 failures on a provider.
#   Assert the provider's cooldown_until is set ~5 minutes in the future.
#   Assert the provider is skipped on subsequent calls within the cooldown window.

# Test: cooled-down provider re-enters rotation after 5 minutes
#   Trigger cooldown on a provider, then advance time past cooldown.
#   Assert the provider is tried again on the next call.
#   Assert consecutive_failures is reset to 0.

# ---- Non-retryable errors ----

# Test: non-retryable error (400) fails immediately without retry
#   Simulate a 400 error.
#   Assert the provider was called exactly once (no retry).
#   Assert failover to next provider happens immediately.

# Test: non-retryable error (401) fails immediately without retry
#   Same as 400 test. Verifies credential errors are not retried.

# ---- Provider chain behavior ----

# Test: with_fallbacks chain exercises providers in correct order
#   Configure 3 providers: A fails (cooldown), B fails (cooldown), C succeeds.
#   Assert calls went A -> B -> C in that order.

# Test: health state is per-provider (provider A failure doesn't affect B)
#   Trigger failures on provider A until cooldown.
#   Assert provider B has consecutive_failures == 0 and no cooldown.
#   Assert provider B is used successfully on next call.
```

## Implementation Details

### File: `src/quantstack/llm/circuit_breaker.py` (new)

This is the core new file. It contains:

**`ProviderHealth` dataclass** -- Tracks per-provider state:
- `consecutive_failures: int = 0`
- `cooldown_until: float | None = None` (monotonic timestamp)
- `last_error: str | None = None`

**`CircuitBreaker` class** -- Main class with the following interface:

```python
class CircuitBreaker:
    """LLM provider circuit breaker with per-provider health tracking.

    Wraps LLM invocations to detect failures, retry with backoff,
    and fail over to alternate providers when a provider is unhealthy.

    Health state is in-memory only. Resets on process restart.
    """

    FAILURE_THRESHOLD: int = 3
    COOLDOWN_SECONDS: float = 300.0  # 5 minutes
    RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503})
    NON_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({400, 401, 403})
    MAX_RETRIES: int = 2
    BACKOFF_BASE: float = 1.0  # seconds; doubles each retry

    def __init__(self, fallback_order: list[str] | None = None):
        """Initialize with provider fallback order.

        Args:
            fallback_order: Provider priority list. Defaults to FALLBACK_ORDER
                from provider.py.
        """
        ...

    def is_available(self, provider: str) -> bool:
        """Check if a provider is available (not in cooldown, under failure threshold)."""
        ...

    def record_success(self, provider: str) -> None:
        """Reset failure count for a provider after successful invocation."""
        ...

    def record_failure(self, provider: str, error: Exception) -> None:
        """Increment failure count. Enter cooldown if threshold reached."""
        ...

    def get_available_providers(self) -> list[str]:
        """Return ordered list of providers not currently in cooldown."""
        ...

    async def invoke_with_failover(self, tier: str, messages: list, **kwargs):
        """Invoke an LLM call with automatic retry and failover.

        Tries the primary provider first. On retryable errors, retries with
        exponential backoff. After FAILURE_THRESHOLD consecutive failures,
        fails over to the next available provider.

        Args:
            tier: LLM tier (heavy/medium/light/bulk).
            messages: LangChain message list.
            **kwargs: Additional kwargs passed to the chat model.

        Returns:
            LLM response from the first successful provider.

        Raises:
            AllProvidersFailedError: Every provider in the chain is
                either in cooldown or failed.
        """
        ...
```

**Key implementation notes:**

- Use `time.monotonic()` for cooldown tracking (immune to wall clock changes).
- The `_classify_error()` helper inspects the exception to extract HTTP status codes. LangChain provider exceptions vary in structure -- check for `status_code` attribute, `response.status_code`, or parse from error message string as fallback.
- Thread safety: Use a `threading.Lock` around health state mutations if the circuit breaker is shared across async tasks. In practice, LangGraph nodes run sequentially within a graph, but multiple graphs share the same process.
- Log at WARNING level when entering cooldown, at INFO when a provider recovers from cooldown.

### File: `src/quantstack/llm/provider.py` (modify)

Integrate the circuit breaker into the existing `get_chat_model()` function. The change is small: after `get_chat_model()` constructs the primary model, wrap it with the circuit breaker's fallback chain.

Two approaches (choose during implementation):

**Option A -- Wrap at `get_chat_model()` level:** Return a model that internally uses `CircuitBreaker.invoke_with_failover()`. This requires a thin wrapper class that implements the LangChain `BaseChatModel` interface (specifically `_generate` and `_agenerate`), delegating to the circuit breaker. This is cleaner but more code.

**Option B -- Wrap at call sites using `with_fallbacks()`:** Build a LangChain fallback chain from available providers and return it from `get_chat_model()`. The circuit breaker acts as a filter that removes cooled-down providers from the chain before construction. This is simpler but means the fallback chain is rebuilt on each `get_chat_model()` call. Since `get_chat_model()` is called once per agent node (not per token), this overhead is negligible.

**Recommended: Option B.** It uses LangChain's built-in `with_fallbacks()` and avoids implementing a custom `BaseChatModel` subclass.

The modification to `get_chat_model()`:

```python
# At the end of get_chat_model(), before returning:

# Build fallback chain from available providers
# (circuit_breaker is a module-level singleton)
from quantstack.llm.circuit_breaker import get_circuit_breaker

cb = get_circuit_breaker()
available = cb.get_available_providers()

# Create fallback models for available providers (excluding primary)
fallback_models = []
for provider_name in available:
    if provider_name == provider:
        continue  # skip primary, it's already the main model
    try:
        # ... instantiate fallback model for this provider/tier
        fallback_models.append(fallback_model)
    except Exception:
        continue  # provider can't be instantiated, skip

if fallback_models:
    return primary_model.with_fallbacks(fallback_models)
return primary_model
```

Additionally, add a module-level `get_circuit_breaker()` function in `circuit_breaker.py` that returns a singleton instance. This ensures all call sites share the same health state.

### Error Classification

The circuit breaker must distinguish retryable from non-retryable errors. LangChain wraps provider-specific exceptions, so the classification logic needs to handle multiple exception types:

- **`httpx.HTTPStatusError`**: Check `response.status_code`
- **`openai.RateLimitError`** / **`anthropic.RateLimitError`**: Always retryable (429)
- **`openai.AuthenticationError`** / **`anthropic.AuthenticationError`**: Never retryable (401)
- **`openai.BadRequestError`**: Never retryable (400)
- **`asyncio.TimeoutError`** / **`httpx.ReadTimeout`**: Retryable (transient network)
- **Generic `Exception`**: Default to retryable (fail safe -- better to retry than crash)

Implement this as a `_classify_error(exc: Exception) -> tuple[bool, int | None]` method that returns `(is_retryable, status_code_or_none)`.

## Verification

After implementation, verify by:

1. **Unit tests pass**: All 9 test cases in `tests/llm/test_circuit_breaker.py`.
2. **Manual smoke test**: Set `LLM_PROVIDER=anthropic`, revoke the API key temporarily, run a graph cycle. Confirm it fails over to the next available provider (e.g., Bedrock) within ~5 seconds and logs a warning.
3. **Langfuse traces**: After deployment, check that provider switches are visible in traces (the model name changes mid-session when failover occurs).

## Rollback

Remove `circuit_breaker.py` and revert the `provider.py` changes. The system returns to startup-only validation with no runtime failover. This is safe because the circuit breaker only adds resilience -- removing it restores the previous crash-on-failure behavior which, while worse, is functionally correct (the graph cycle retries on the next scheduler tick).
