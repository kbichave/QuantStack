# Section 06: Prompt Caching

## Overview

This section enables Anthropic prompt caching across all LLM calls in the QuantStack system. Without prompt caching, the Haiku-tier strategy agents running 78 cycles/day during market hours will exceed the $10/day token budget on system prompt tokens alone. Prompt caching reduces input token costs by 80-90% for repeated content (system prompts, tool definitions, strategy context) that remains stable across invocations.

**No dependencies.** This section can be implemented independently of all other Phase 10 sections. It blocks section-12 (Governance), which relies on cached prompts to hit the ~$4.35/day operational cost target for the CIO + strategy agent hierarchy.

## Background

### Current LLM Architecture

The LLM layer lives in two files:

- `src/quantstack/llm/config.py` -- Defines `ProviderConfig` (model IDs per tier), `ModelConfig` (instantiation params), `PROVIDER_CONFIGS` dict, and `VALID_TIERS`.
- `src/quantstack/llm/provider.py` -- `get_model()`, `get_model_with_fallback()`, `get_model_for_role()`, and `get_chat_model()` resolve tiers to model strings and instantiate LangChain `ChatModel` objects.

The primary provider is Bedrock (`ChatBedrock` from `langchain_aws`). Fallback chain: bedrock -> anthropic -> openai -> groq -> ollama. The `get_chat_model()` function is the main entry point for graph nodes; it returns a configured `ChatBedrock` or `ChatAnthropic` instance.

### How Prompt Caching Works

Anthropic's prompt caching lets you mark portions of a prompt (system instructions, tool definitions) with `cache_control={"type": "ephemeral"}`. On subsequent requests within the cache TTL (currently 5 minutes, extended on each cache hit), cached tokens are billed at 10% of input token price. The first request pays a small write premium (25% above input price), but all subsequent hits within the TTL save 90%.

For Bedrock, caching is enabled via the `anthropic_beta` header value `prompt-caching-2024-07-31`. For direct Anthropic API, it is enabled automatically when `cache_control` markers are present.

### Minimum Cacheable Sizes

- Claude Opus: 1,024 tokens (was 4,096, reduced in later API versions)
- Claude Sonnet: 1,024 tokens (was 2,048, reduced)
- Claude Haiku: 1,024 tokens (was 2,048, reduced)

All agent system prompts in `agents.yaml` exceed these minimums. Tool definition blocks (100+ tools) are well above any minimum.

### Cost Impact

Without caching (Haiku strategy agents, 78 cycles/day):
- ~4K tokens system prompt per cycle, ~$0.015/cycle = ~$1.17/day just for strategy agents

With caching (80% hit rate conservatively):
- ~$0.003/cycle = ~$0.23/day for strategy agents
- Total daily savings: $1-3/day on strategy agents, more on research cycles
- Critical for hitting the $10/day combined target (operational + overnight research)

## Tests First

File: `tests/unit/test_prompt_caching.py`

```python
"""Tests for prompt caching configuration in LLM provider layer."""

import os
from unittest.mock import MagicMock, patch


def test_cache_control_added_to_bedrock_calls():
    """get_chat_model for Bedrock tiers returns a ChatBedrock instance
    configured with the prompt-caching beta header so that downstream
    system-prompt and tool-definition messages can use cache_control markers.

    Verify that the ChatBedrock constructor receives the beta header
    'prompt-caching-2024-07-31' in its model_kwargs or beta_use_converse_api
    configuration.
    """


def test_system_prompt_includes_cache_control_marker():
    """When building agent messages, the system prompt message includes
    cache_control={'type': 'ephemeral'} so the Anthropic API caches it.

    This tests the helper that wraps system prompt content with the
    cache_control marker before passing to the model.
    """


def test_tool_definitions_include_cache_control_marker():
    """Tool definition blocks bound to an agent include cache_control
    markers so repeated cycles don't re-process the full tool schema.

    Verify that the last tool in the tool list carries the
    cache_control={'type': 'ephemeral'} marker (Anthropic caches
    everything up to and including the marked block).
    """


def test_cache_invalidation_on_tool_definition_change():
    """When tool definitions change (version bump in tool_manifest.yaml
    or TOOL_ADDED event), the cache is effectively invalidated because
    the content hash changes. Verify that a changed tool list produces
    a different prompt payload (no stale cache hit).

    This is a behavioral property test -- prompt caching invalidation
    is handled server-side by content hash, so the test verifies that
    modified tool definitions produce different message content.
    """


def test_caching_disabled_via_env_var():
    """When PROMPT_CACHING_ENABLED=false, get_chat_model returns a model
    without any cache_control configuration. This is the escape hatch
    for debugging or cost comparison.
    """
```

## Implementation Details

### Files to Modify

1. **`src/quantstack/llm/config.py`** -- Add caching configuration constants.
2. **`src/quantstack/llm/provider.py`** -- Add `cache_control` support to `get_chat_model()` and `_instantiate_chat_model()`.

No new files are created. This is a modification of the existing LLM layer.

### Changes to `src/quantstack/llm/config.py`

Add a caching configuration section:

```python
# --- Prompt caching ---

# Feature flag: set PROMPT_CACHING_ENABLED=false to disable
PROMPT_CACHING_ENABLED_DEFAULT = True

# Bedrock beta header required to enable prompt caching
BEDROCK_PROMPT_CACHING_BETA = "prompt-caching-2024-07-31"

# Cache control marker applied to system prompts and tool definitions
CACHE_CONTROL_EPHEMERAL = {"type": "ephemeral"}
```

These are constants only -- no behavioral logic in config.py.

### Changes to `src/quantstack/llm/provider.py`

#### 1. Add `prompt_caching` parameter to `get_chat_model()`

Extend the function signature:

```python
def get_chat_model(
    tier: str,
    thinking: dict | None = None,
    temperature: float | None = None,
    prompt_caching: bool | None = None,
):
```

When `prompt_caching` is `None` (default), read from `PROMPT_CACHING_ENABLED` env var (default `True`). When explicitly `True` or `False`, use that value. Pass the resolved boolean through to `_instantiate_chat_model()` via a new field on `ModelConfig`.

#### 2. Add `prompt_caching` field to `ModelConfig`

```python
@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_id: str
    tier: str
    max_tokens: int = 4096
    temperature: float = 0.0
    thinking: dict | None = None
    prompt_caching: bool = False
```

#### 3. Modify `_instantiate_chat_model()` for Bedrock

In the Bedrock branch, when `config.prompt_caching` is True, add the beta header:

```python
if provider == "bedrock":
    kwargs = {
        "model_id": config.model_id,
        "region_name": os.environ.get("AWS_DEFAULT_REGION", os.environ.get("BEDROCK_REGION", "us-east-1")),
        "model_kwargs": {"temperature": config.temperature, "max_tokens": config.max_tokens},
    }
    if config.prompt_caching:
        # Enable Anthropic prompt caching via Bedrock's beta header.
        # This allows system prompts and tool definitions marked with
        # cache_control={"type": "ephemeral"} to be cached server-side.
        kwargs["beta_use_converse_api"] = False  # Use Messages API for caching support
        kwargs["model_kwargs"]["anthropic_beta"] = [BEDROCK_PROMPT_CACHING_BETA]
    return ChatBedrock(**kwargs)
```

Note: The exact kwarg name depends on the `langchain_aws` version. `ChatBedrockConverse` uses different kwargs than `ChatBedrock`. Check the installed version. If using `ChatBedrockConverse`, the beta header goes in `additional_model_request_fields`. If using the raw `ChatBedrock` with the Messages API, it goes in `model_kwargs`. The test should verify the header is present regardless of the specific kwarg path.

#### 4. Modify `_instantiate_chat_model()` for Anthropic (direct API)

In the Anthropic branch, when `config.prompt_caching` is True, add the beta header:

```python
if provider == "anthropic":
    kwargs = {
        "model": config.model_id,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.prompt_caching:
        # Direct Anthropic API: caching is enabled by default when
        # cache_control markers are present in messages. The model_kwargs
        # or headers may need the beta flag for older SDK versions.
        kwargs["model_kwargs"] = kwargs.get("model_kwargs", {})
        kwargs["model_kwargs"]["extra_headers"] = {
            "anthropic-beta": BEDROCK_PROMPT_CACHING_BETA
        }
    if config.thinking:
        # ... existing thinking config ...
    return ChatAnthropic(**kwargs)
```

#### 5. Non-Anthropic providers

For OpenAI, Gemini, Ollama, and Groq: `prompt_caching` is silently ignored. These providers either handle caching internally or do not support it. Log a debug message if `prompt_caching=True` for a non-Anthropic provider.

### How Cache Markers Are Applied at Call Sites

The changes above enable the *infrastructure* for caching. The actual cache markers (`cache_control={"type": "ephemeral"}`) are applied at the **call site** where messages are constructed -- specifically in graph nodes that build the system prompt and tool binding. This section covers only the provider-layer enablement.

For reference, the call-site pattern (implemented by graph node code, not this section) looks like:

```python
from quantstack.llm.config import CACHE_CONTROL_EPHEMERAL

system_message = SystemMessage(
    content="You are a swing trading agent...",
    additional_kwargs={"cache_control": CACHE_CONTROL_EPHEMERAL},
)
```

And for tool definitions, the last tool in the bound set carries the marker so the entire tool block is cached:

```python
tools = get_tools_for_agent(agent_name)
if tools and prompt_caching_enabled:
    # Mark last tool for caching -- Anthropic caches everything
    # up to and including the marked element
    tools[-1].cache_control = CACHE_CONTROL_EPHEMERAL
```

These call-site changes are small and localized. They belong in the graph node code or tool-binding layer, not in the LLM provider module. This section ensures the provider is configured to *honor* those markers.

### Keeping Tool Definitions Stable

Cache invalidation is content-hash-based on the Anthropic side. Any change to tool definitions (adding a tool, modifying a description) invalidates the cache for all agents using that tool set. To maximize cache hit rate:

- Tool additions happen only via event bus notifications (`TOOL_ADDED`), not mid-cycle.
- `tool_manifest.yaml` (from section-02) should be versioned. Tool binding is rebuilt only when the manifest version changes.
- Agent tool sets are resolved at cycle start and held constant for the cycle duration.

This is an operational discipline enforced by the tool lifecycle (section-02), not by code in this section.

### Environment Variable

- `PROMPT_CACHING_ENABLED` -- Set to `"false"` to disable all caching configuration. Default: `"true"`. Useful for cost comparison testing or debugging cache-related issues.

## Verification Checklist

1. `get_chat_model("light", prompt_caching=True)` with Bedrock provider returns a `ChatBedrock` instance with the prompt-caching beta header present.
2. `get_chat_model("heavy", prompt_caching=True)` with Anthropic provider returns a `ChatAnthropic` with the appropriate headers.
3. `get_chat_model("light", prompt_caching=False)` returns an instance without any caching headers.
4. `get_chat_model("light")` with `PROMPT_CACHING_ENABLED=false` in env returns an instance without caching headers.
5. `get_chat_model("light")` with `PROMPT_CACHING_ENABLED=true` (or unset) returns an instance with caching headers.
6. Non-Anthropic providers (openai, ollama, groq) silently ignore `prompt_caching=True`.
7. All existing tests in the LLM provider module continue to pass (backward compatible -- default caching=True is additive, not breaking).

## Dependencies and Interactions

- **Depends on:** Nothing. This is a standalone infrastructure change.
- **Blocks:** Section-12 (Governance) relies on cached prompts to hit the $4.35/day operational cost target for CIO + Haiku strategy agents.
- **Interacts with:** Section-02 (Tool Lifecycle) -- stable tool definitions maximize cache hit rate. Section-14 (AutoResearchClaw Upgrades) -- tool_implement changes tool definitions, which invalidates cache.
