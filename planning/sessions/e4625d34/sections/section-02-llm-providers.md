# Section 02: LLM Provider Management

## Overview

This section implements the LLM provider abstraction layer that all CrewAI agents use to select the correct model for their reasoning tier. It consists of two modules:

- `src/quantstack/llm/provider.py` — Provider selection function with fallback chain
- `src/quantstack/llm/config.py` — Model tier map per provider (dataclass + registry)

The module does NOT wrap CrewAI's `LLM` class. It provides configuration strings that CrewAI agents consume via their `llm` parameter. CrewAI handles the actual LLM client instantiation internally.

**Dependencies:** Section 01 (project scaffolding) must be complete — specifically the directory structure and `pyproject.toml` with CrewAI dependencies.

**Blocks:** Sections 04 (agent definitions), 05 (crew workflows), and 09 (runners) all depend on this module to resolve model strings at crew instantiation time.

---

## Tests (Write These First)

File: `tests/unit/test_llm_provider.py`

```python
# --- config.py tests ---

# Test: get_model("heavy") returns correct model string for each provider (bedrock, anthropic, openai, gemini, ollama)
# Test: get_model("medium") returns correct model string per provider
# Test: get_model("light") returns correct model string per provider
# Test: get_model with unknown tier raises ValueError
# Test: LLM_PROVIDER env var selects primary provider
# Test: LLM_PROVIDER defaults to "bedrock" when unset

# --- provider.py fallback tests ---

# Test: fallback chain activates when primary provider raises exception
# Test: fallback chain tries providers in order: bedrock -> anthropic -> openai -> ollama
# Test: fallback chain raises after all providers fail
# Test: provider config validates required env vars per provider (e.g., AWS keys for bedrock)
```

### Test Details

**`get_model` tier tests:** For each of the 5 providers, set `LLM_PROVIDER` env var, call `get_model("heavy")`, and assert the returned string matches the expected model identifier from the tier map. Repeat for `"medium"`, `"light"`, and `"embedding"`.

**Unknown tier test:** Call `get_model("nonexistent")` and assert it raises `ValueError`.

**Default provider test:** Ensure `LLM_PROVIDER` is unset (use `monkeypatch.delenv`), call `get_model("heavy")`, and assert it returns the Bedrock heavy model string.

**Fallback chain tests:** Mock the LLM validation/health check to raise a provider-specific error for the primary provider. Assert that `get_model_with_fallback` (or equivalent) returns the next provider's model string. Chain through all providers and verify the order is bedrock -> anthropic -> openai -> ollama. When all providers fail, assert the function raises an appropriate exception (e.g., `AllProvidersFailedError`).

**Env var validation test:** For the `bedrock` provider, unset `AWS_ACCESS_KEY_ID`. Assert that requesting a bedrock model raises a configuration error indicating the missing credential. Repeat for other providers with their respective required env vars.

---

## Implementation Details

### File: `src/quantstack/llm/__init__.py`

Empty init file to make `llm` a package. Optionally re-export `get_model` for convenience:

```python
from quantstack.llm.provider import get_model
```

### File: `src/quantstack/llm/config.py`

Defines a `ProviderConfig` dataclass and a registry mapping provider names to their tier configurations.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ProviderConfig:
    """Model identifiers for each reasoning tier within a single LLM provider.

    Tiers:
        heavy  — fund-manager, quant-researcher, trade-debater, risk, ml-scientist,
                 strategy-rd, options-analyst
        medium — earnings-analyst, position-monitor, daily-planner, market-intel,
                 trade-reflector
        light  — community-intel, execution-researcher, supervisor
        embedding — memory, RAG
    """
    heavy: str
    medium: str
    light: str
    embedding: str
```

The provider registry is a module-level dict mapping provider name strings to `ProviderConfig` instances:

| Provider | Heavy | Medium | Light | Embedding |
|----------|-------|--------|-------|-----------|
| `bedrock` | `bedrock/anthropic.claude-sonnet-4-20250514-v1:0` | `bedrock/anthropic.claude-sonnet-4-20250514-v1:0` | `bedrock/anthropic.claude-haiku-4-5-20251001-v1:0` | `ollama/mxbai-embed-large` |
| `anthropic` | `anthropic/claude-sonnet-4` | `anthropic/claude-sonnet-4` | `anthropic/claude-haiku-4-5` | `ollama/mxbai-embed-large` |
| `openai` | `openai/gpt-4o` | `openai/gpt-4o-mini` | `openai/gpt-4o-mini` | `openai/text-embedding-3-small` |
| `gemini` | `gemini/gemini-2.5-pro` | `gemini/gemini-2.0-flash` | `gemini/gemini-2.0-flash` | `ollama/mxbai-embed-large` |
| `ollama` | `ollama/llama3:70b` | `ollama/llama3.2` | `ollama/llama3.2` | `ollama/mxbai-embed-large` |

The registry dict is named `PROVIDER_CONFIGS` and uses string keys matching the `LLM_PROVIDER` env var values.

Valid tiers are: `"heavy"`, `"medium"`, `"light"`, `"embedding"`. Accessing any other tier raises `ValueError`.

Each provider also has required environment variables for validation:

| Provider | Required Env Vars |
|----------|-------------------|
| `bedrock` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `gemini` | `GEMINI_API_KEY` |
| `ollama` | `OLLAMA_BASE_URL` |

Store these as a dict mapping provider name to a list of required env var names. The validation function checks that all required vars are set and non-empty for a given provider.

### File: `src/quantstack/llm/provider.py`

This is the main interface module. It exposes:

```python
def get_model(tier: str) -> str:
    """Return the model string for CrewAI Agent's llm parameter.

    Reads LLM_PROVIDER from environment (default: "bedrock").
    Validates that required env vars for the provider are set.
    Returns the model identifier string for the requested tier.

    Args:
        tier: One of "heavy", "medium", "light", "embedding".

    Raises:
        ValueError: If tier is not recognized.
        ConfigurationError: If required env vars for the provider are missing.
        KeyError: If LLM_PROVIDER names an unknown provider.
    """
```

The function:
1. Reads `LLM_PROVIDER` from `os.environ`, defaulting to `"bedrock"`.
2. Looks up the provider in `PROVIDER_CONFIGS`. If not found, raises `KeyError`.
3. Validates that all required env vars for that provider are present. If not, raises a `ConfigurationError` (custom exception, defined in this module or `config.py`).
4. Returns `getattr(provider_config, tier)`. If `tier` is not a valid field, raises `ValueError`.

### Fallback Chain

The fallback chain order is: **bedrock -> anthropic -> openai -> ollama**.

```python
FALLBACK_ORDER: list[str] = ["bedrock", "anthropic", "openai", "ollama"]

def get_model_with_fallback(tier: str) -> str:
    """Return model string, falling back through providers on failure.

    Tries the primary provider first (from LLM_PROVIDER env var).
    If it fails validation or health check, tries the next provider
    in the fallback chain.

    Only active when LLM_FALLBACK_ENABLED=true (default: true).
    If fallback is disabled, behaves identically to get_model().

    Raises:
        AllProvidersFailedError: When every provider in the chain fails.
    """
```

The fallback function:
1. Checks `LLM_FALLBACK_ENABLED` env var (default `"true"`). If `"false"`, delegates to `get_model()` directly.
2. Builds the provider attempt order: primary provider first, then remaining providers from `FALLBACK_ORDER` (skipping the primary since it was already tried).
3. For each provider, attempts to validate env vars and return the model string.
4. If a provider fails validation (missing env vars), logs a warning and continues to the next.
5. If all providers fail, raises `AllProvidersFailedError` with details of each failure.

### Custom Exceptions

Define in `provider.py` (or `config.py`):

```python
class ConfigurationError(Exception):
    """Raised when required environment variables are missing for a provider."""

class AllProvidersFailedError(Exception):
    """Raised when the entire fallback chain is exhausted."""
```

### Environment Variables

The module reads:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `"bedrock"` | Primary LLM provider name |
| `LLM_FALLBACK_ENABLED` | `"true"` | Enable fallback chain |
| `AWS_ACCESS_KEY_ID` | — | Required for bedrock |
| `AWS_SECRET_ACCESS_KEY` | — | Required for bedrock |
| `AWS_DEFAULT_REGION` | `"us-east-1"` | Required for bedrock |
| `ANTHROPIC_API_KEY` | — | Required for anthropic |
| `OPENAI_API_KEY` | — | Required for openai |
| `GEMINI_API_KEY` | — | Required for gemini |
| `OLLAMA_BASE_URL` | `"http://ollama:11434"` | Required for ollama |

### Usage by Downstream Sections

Crew definitions (Section 04) use this module at instantiation time to inject model strings into agent YAML variables:

```python
from quantstack.llm.provider import get_model

# When building a crew:
crew = TradingCrew()
crew.kickoff(inputs={
    "heavy_model": get_model("heavy"),
    "medium_model": get_model("medium"),
    "light_model": get_model("light"),
})
```

Agent YAML references these as `{heavy_model}`, `{medium_model}`, `{light_model}` in their `llm` field.

The runner modules (Section 09) call `get_model_with_fallback()` instead of `get_model()` to enable automatic provider switching during runtime failures.

---

## Design Decisions

**Why not wrap CrewAI's LLM class:** CrewAI already handles LLM client lifecycle, retry logic, and streaming. Wrapping it would create a leaky abstraction. Instead, this module only resolves the model string — CrewAI does the rest.

**Why validation at call time, not import time:** The module is imported during crew construction. Validating at call time (not module load) allows tests to run without setting all env vars. It also enables the fallback chain to skip providers with missing credentials gracefully.

**Why frozen dataclass:** The tier map is immutable configuration. A frozen dataclass prevents accidental mutation and makes it hashable for caching if needed later.

**Why ollama is the last fallback:** Ollama runs locally in Docker and requires no API keys, so it should always be available as a last resort. The tradeoff is lower reasoning quality (llama3.2 vs. Claude Sonnet), but it keeps the system running during external API outages.
