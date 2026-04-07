# Section 07: Remove Hardcoded Model Strings

**Plan Reference:** Item 5.6
**Dependencies:** section-01 (consolidated LLM config)
**Blocks:** section-08-litellm-proxy

---

## Problem

6+ files hardcode provider-specific model strings instead of using `get_chat_model()`. Prevents global provider switching, blocks LiteLLM deployment.

## Tests (Write First)

Create `tests/unit/test_no_hardcoded_models.py`:

```python
# Test: grep src/quantstack/ for hardcoded model patterns returns zero hits
#       (excluding llm/config.py and litellm_config.yaml)
#       Patterns: claude-sonnet, claude-haiku, gpt-4o, llama-3,
#                 anthropic/, bedrock/, openai/, groq/, gemini/

# Test: no direct LLM instantiation (ChatAnthropic, ChatBedrock, etc.)
#       outside llm/provider.py
```

---

## Implementation

### Step 1: Comprehensive Grep

```bash
grep -rn "claude-sonnet\|claude-haiku\|gpt-4o\|llama-3" src/quantstack/ --include="*.py" | grep -v "llm/config.py"
grep -rn "anthropic/\|bedrock/\|openai/\|groq/\|gemini/" src/quantstack/ --include="*.py" | grep -v "llm/config.py"
```

### Step 2: Add `bulk` Tier

For OPRO/TextGrad loops that need cheap high-volume models. Add to `llm/config.py`:
- Primary: `groq/llama-3.3-70b-versatile`
- Fallback: `bedrock/haiku`

### Step 3: Replace Each Hardcoded String

Known locations (verify these still exist):
- `tool_search_compat.py:21` → `get_chat_model("heavy")`
- `trading/nodes.py:843` → `get_chat_model("heavy")`
- `trade_evaluator.py` → `get_chat_model("heavy")`
- `mem0_client.py` → `get_chat_model("medium")` (special: may need provider_hint for OpenAI compat)
- `hypothesis_agent.py` → `get_chat_model("bulk")`
- `opro_loop.py` → `get_chat_model("bulk")`

### Step 4: Update Imports

Remove direct provider imports (`ChatAnthropic`, `ChatBedrock`, etc.) from all files outside `llm/provider.py`.

### Step 5: Validate

Run regression test. Verify Langfuse traces show correct model routing.

---

## Rollback

Not easily reversible (hardcoded strings removed). But `get_chat_model()` can be reconfigured per tier in `llm/config.py` without touching call sites.

## Files

| File | Change |
|------|--------|
| `src/quantstack/llm/config.py` | Modify — add bulk tier |
| Various files with hardcoded strings | Modify — replace with get_chat_model() |
| `tests/unit/test_no_hardcoded_models.py` | **Create** |
