# Section 01: Consolidate Dual LLM Config Systems

**Plan Reference:** Item 5.0 (Prerequisite)
**Dependencies:** None
**Blocks:** section-04-tier-reclassification, section-07-hardcoded-strings, section-08-litellm-proxy

---

## Problem

Two independent LLM routing systems coexist:

1. **`src/quantstack/llm/provider.py` + `src/quantstack/llm/config.py`** — Tier system (heavy/medium/light), used by graph agent executor
2. **`src/quantstack/llm_config.py`** — Legacy tier system (IC/Pod/Assistant/Decoder/Workshop), used by standalone scripts and tools

Each has its own provider fallback chain, env var overrides (`LLM_MODEL_IC`, `LLM_MODEL_POD`, etc.), and model string resolution. Downstream cost optimization items would need to address both systems independently.

---

## Tests (Write First)

Create `tests/unit/test_llm_config_consolidation.py`:

```python
# Test: IC tier maps to "light" and returns same model string
# Test: Pod tier maps to "heavy" and returns same model string
# Test: Workshop tier maps to "heavy" and returns same model string
# Test: Assistant tier maps to "heavy" and returns same model string
# Test: Decoder tier maps to "light" and returns same model string
# Test: env var override LLM_MODEL_IC still works after migration (backward compat)
# Test: no file in src/quantstack/ imports from llm_config after migration (grep-based)
# Test: importing from quantstack.llm_config raises DeprecationWarning after migration
```

---

## Implementation Steps

### Step 1: Audit Current Usage

```bash
grep -rn "from quantstack.llm_config import" src/quantstack/
grep -rn "from quantstack.llm.provider import" src/quantstack/
grep -rn "from quantstack.llm.config import" src/quantstack/
grep -rn "LLM_MODEL_IC\|LLM_MODEL_POD\|LLM_MODEL_ASSISTANT\|LLM_MODEL_DECODER\|LLM_MODEL_WORKSHOP" src/quantstack/
```

### Step 2: Map Legacy Tiers

| Legacy Tier | Current Tier | Rationale |
|-------------|--------------|-----------|
| IC | light | Lightweight extraction |
| Pod | heavy | Heavy reasoning |
| Assistant | heavy | Complex coordination |
| Decoder | light | Fast structured output |
| Workshop | heavy | Strategy synthesis |

### Step 3: Add Tier Aliases to `provider.py`

```python
TIER_ALIASES = {
    "ic": "light",
    "pod": "heavy",
    "assistant": "heavy",
    "decoder": "light",
    "workshop": "heavy",
}
```

Normalize tier name in `get_chat_model()` — case-insensitive, map aliases, log when legacy tier used.

### Step 4: Add Env Var Aliases

Map `LLM_MODEL_IC` → light tier override, `LLM_MODEL_POD` → heavy tier override, etc.

### Step 5: Migrate All Callers

For each file importing from `llm_config.py`, replace with `get_chat_model(tier)`:
- `hypothesis_agent.py` — Workshop → heavy
- `opro_loop.py` — Decoder → light
- Sentiment collectors — IC → light

### Step 6: Deprecate Legacy Config

Add `DeprecationWarning` to `llm_config.py`. Keep redirect stubs for one release cycle.

### Step 7: Remove Legacy Config (Next Release)

Delete `llm_config.py` after all callers migrated.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Model mismatch after migration | Validate tier mappings against actual model strings. Langfuse trace comparison. |
| Missed call sites | Grep-based test catches remaining imports |
| Env var overrides broken | Specific tests for each legacy env var |

---

## Verification Checklist

- [ ] All tests pass
- [ ] `grep -r "from quantstack.llm_config import" src/quantstack/` returns zero hits
- [ ] Langfuse traces show identical model IDs before/after
- [ ] Legacy env vars still work
- [ ] Deprecation warning emitted

## Files Modified

- `src/quantstack/llm/provider.py` — Tier alias normalization
- `src/quantstack/llm/config.py` — Env var override handling
- `src/quantstack/llm_config.py` — Deprecation + redirect stubs
- Caller files (hypothesis_agent.py, opro_loop.py, etc.)
- `tests/unit/test_llm_config_consolidation.py` — New
