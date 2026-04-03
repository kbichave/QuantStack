# Section 02: LLM Provider Refactor

## Purpose

Add a `get_chat_model(tier) -> BaseChatModel` function to the LLM provider module so that LangGraph nodes can receive instantiated LangChain chat model objects instead of plain model-identifier strings. The existing `get_model()` function (returns a string) stays untouched for backward compatibility.

## Dependencies

- **section-01-scaffolding** must be complete (LangGraph and langchain dependencies installed in `pyproject.toml`).
- No other section dependencies.

## Blocked By This Section

Sections 06 (supervisor graph), 07 (research graph), and 08 (trading graph) all call `get_chat_model()` when building graph nodes.

---

## Current State

There are two files in `src/quantstack/llm/`:

**`config.py`** defines a `ProviderConfig` frozen dataclass with four tier fields (`heavy`, `medium`, `light`, `embedding`) and a `PROVIDER_CONFIGS` dict mapping provider names to their tier-specific model identifier strings. It also has `VALID_TIERS` and `REQUIRED_ENV_VARS`.

**`provider.py`** provides:
- `get_model(tier) -> str` -- reads `LLM_PROVIDER` env var (default `"bedrock"`), validates credentials, returns the model string for the requested tier.
- `get_model_with_fallback(tier) -> str` -- tries the primary provider, then walks `FALLBACK_ORDER` until one validates.
- Custom exceptions: `ConfigurationError`, `AllProvidersFailedError`.

**`__init__.py`** re-exports `get_model` and `get_model_with_fallback`.

The provider config uses a `"provider/model-id"` string format (e.g., `"bedrock/anthropic.claude-sonnet-4-20250514-v1:0"`). The prefix before `/` identifies which LangChain ChatModel class to instantiate.

---

## Implementation

### 2.1 Add `ModelConfig` dataclass to `config.py`

Add a frozen dataclass that captures all parameters needed to instantiate a ChatModel:

```python
@dataclass(frozen=True)
class ModelConfig:
    provider: str       # "bedrock", "anthropic", "openai", "gemini", "ollama"
    model_id: str       # full model identifier string from ProviderConfig
    tier: str           # "heavy", "medium", "light"
    max_tokens: int = 4096
    temperature: float = 0.0
```

This dataclass is consumed by `get_chat_model()` to select the right ChatModel class and pass constructor arguments. The `frozen=True` ensures configs are not accidentally mutated after creation.

### 2.2 Add `get_chat_model()` to `provider.py`

Add a new function alongside the existing ones. Do not modify `get_model()` or `get_model_with_fallback()`.

**Signature:**

```python
def get_chat_model(tier: str) -> BaseChatModel:
    """Return a configured LangChain ChatModel for the given tier.

    Uses the same provider resolution and fallback logic as get_model_with_fallback().
    Parses the provider prefix from the model string to select the ChatModel class.
    """
```

**Logic:**

1. Call `get_model_with_fallback(tier)` to get the model identifier string (reuses all existing provider selection, validation, and fallback logic).
2. Parse the provider prefix from the string (everything before the first `/`).
3. Extract the model ID (everything after the first `/`).
4. Build a `ModelConfig` from the parsed values plus tier defaults.
5. Instantiate the appropriate ChatModel class via a provider-to-class dispatch dict.

**Provider dispatch mapping:**

| Provider prefix | ChatModel class | Import path | Key constructor args |
|-----------------|----------------|-------------|---------------------|
| `bedrock` | `ChatBedrock` | `langchain_aws` | `model_id`, `region_name` (from `AWS_DEFAULT_REGION`), `model_kwargs={"temperature": ..., "max_tokens": ...}` |
| `anthropic` | `ChatAnthropic` | `langchain_anthropic` | `model`, `temperature`, `max_tokens` |
| `openai` | `ChatOpenAI` | `langchain_openai` | `model`, `temperature`, `max_tokens` |
| `gemini` | `ChatGoogleGenerativeAI` | `langchain_google_genai` | `model`, `temperature`, `max_output_tokens` |
| `ollama` | `ChatOllama` | `langchain_ollama` | `model`, `temperature` |

**Important design decisions:**

- The imports for provider-specific ChatModel classes should be deferred (inside the dispatch function) rather than at module level. This is a legitimate exception to the "all imports at top" rule because not all providers will be installed in every environment. A `bedrock`-only deployment should not fail at import time because `langchain_openai` is not installed. Guard each import with a try/except that raises a clear error: `"Provider 'openai' requires langchain-openai. Install it with: pip install langchain-openai"`.
- `temperature=0.0` is the right default for a trading system -- deterministic outputs for reproducible decisions.
- The `embedding` tier is intentionally excluded from `get_chat_model()`. Embedding models are not chat models. If someone passes `tier="embedding"`, raise `ValueError` with a clear message directing them to the embedding-specific interface in `rag/embeddings.py`.

### 2.3 Add fallback-aware variant

Since `get_chat_model()` internally calls `get_model_with_fallback()`, the fallback chain is already wired in. There is no need for a separate `get_chat_model_with_fallback()`. This keeps the API surface small.

### 2.4 Update `__init__.py`

Add `get_chat_model` and `ModelConfig` to the re-exports in `src/quantstack/llm/__init__.py`.

### 2.5 Files to modify

| File | Action |
|------|--------|
| `src/quantstack/llm/config.py` | Add `ModelConfig` dataclass |
| `src/quantstack/llm/provider.py` | Add `get_chat_model()` function and provider dispatch |
| `src/quantstack/llm/__init__.py` | Add re-exports for `get_chat_model`, `ModelConfig` |
| `tests/unit/test_llm_provider.py` | Add new test class `TestGetChatModel` |

---

## Tests (Write Before Implementation)

Add these tests to `tests/unit/test_llm_provider.py`. They validate the new `get_chat_model()` function and `ModelConfig` dataclass while leaving existing tests untouched.

### TestModelConfig

```python
class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_model_config_is_frozen(self):
        """ModelConfig instances are immutable after creation."""
        from quantstack.llm.config import ModelConfig
        cfg = ModelConfig(provider="anthropic", model_id="claude-sonnet-4", tier="heavy")
        with pytest.raises(AttributeError):
            cfg.provider = "openai"

    def test_model_config_defaults(self):
        """ModelConfig applies correct defaults for max_tokens and temperature."""
        from quantstack.llm.config import ModelConfig
        cfg = ModelConfig(provider="anthropic", model_id="claude-sonnet-4", tier="heavy")
        assert cfg.max_tokens == 4096
        assert cfg.temperature == 0.0
```

### TestGetChatModel

```python
class TestGetChatModel:
    """Tests for get_chat_model() → BaseChatModel."""

    def test_returns_base_chat_model_heavy(self, monkeypatch):
        """get_chat_model('heavy') returns a BaseChatModel instance."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from langchain_core.language_models import BaseChatModel
        from quantstack.llm.provider import get_chat_model
        model = get_chat_model("heavy")
        assert isinstance(model, BaseChatModel)

    def test_returns_base_chat_model_medium(self, monkeypatch):
        """get_chat_model('medium') returns a BaseChatModel instance."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from langchain_core.language_models import BaseChatModel
        from quantstack.llm.provider import get_chat_model
        model = get_chat_model("medium")
        assert isinstance(model, BaseChatModel)

    def test_returns_base_chat_model_light(self, monkeypatch):
        """get_chat_model('light') returns a BaseChatModel instance."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from langchain_core.language_models import BaseChatModel
        from quantstack.llm.provider import get_chat_model
        model = get_chat_model("light")
        assert isinstance(model, BaseChatModel)

    def test_invalid_tier_raises_value_error(self, monkeypatch):
        """get_chat_model('invalid') raises ValueError."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_chat_model
        with pytest.raises(ValueError, match="tier"):
            get_chat_model("invalid")

    def test_embedding_tier_raises_value_error(self, monkeypatch):
        """get_chat_model('embedding') raises ValueError -- embeddings are not chat models."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_chat_model
        with pytest.raises(ValueError, match="embedding"):
            get_chat_model("embedding")

    def test_anthropic_provider_returns_chat_anthropic(self, monkeypatch):
        """When LLM_PROVIDER=anthropic, returns ChatAnthropic."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_chat_model
        model = get_chat_model("heavy")
        assert type(model).__name__ == "ChatAnthropic"

    def test_fallback_applies_to_chat_model(self, monkeypatch):
        """Fallback chain works: bedrock creds missing, falls back to anthropic ChatModel."""
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from langchain_core.language_models import BaseChatModel
        from quantstack.llm.provider import get_chat_model
        model = get_chat_model("heavy")
        assert isinstance(model, BaseChatModel)
        assert type(model).__name__ == "ChatAnthropic"

    def test_get_model_still_returns_string(self, monkeypatch):
        """Backward compat: get_model() still returns a string, not a ChatModel."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_model
        result = get_model("heavy")
        assert isinstance(result, str)
```

---

## Edge Cases and Failure Modes

1. **Missing provider package**: If `langchain-anthropic` is not installed but `LLM_PROVIDER=anthropic`, `get_chat_model()` should raise an `ImportError` with a clear message naming the missing package and the pip install command. This is distinct from `ConfigurationError` (missing env vars) -- it is a deployment misconfiguration, not a runtime credential issue.

2. **Provider prefix mismatch**: The model string `"bedrock/anthropic.claude-sonnet-4-..."` has `bedrock` as the prefix. The dispatch must use this prefix, not the `LLM_PROVIDER` env var, because after fallback the actual provider used may differ from the configured primary.

3. **Ollama connectivity**: `ChatOllama` will not fail at instantiation if the Ollama server is down -- it fails on first `.invoke()`. The `get_chat_model()` function only validates env vars (that `OLLAMA_BASE_URL` is set), not connectivity. This is intentional: connectivity is a runtime concern, not a config concern.

4. **Thread safety**: `get_chat_model()` creates a new ChatModel instance on every call. This is safe for concurrent use. Do not cache/singleton the ChatModel -- LangGraph may invoke multiple nodes concurrently, each needing its own model instance to avoid shared mutable state.

---

## Verification

After implementation, confirm:

1. `uv run pytest tests/unit/test_llm_provider.py -v` passes all existing tests (no regressions) and all new tests.
2. `get_model("heavy")` still returns a string (backward compat preserved).
3. `get_chat_model("heavy")` returns a `BaseChatModel` instance.
4. `get_chat_model("embedding")` raises `ValueError`.
5. Fallback chain works for chat models the same way it does for strings.
