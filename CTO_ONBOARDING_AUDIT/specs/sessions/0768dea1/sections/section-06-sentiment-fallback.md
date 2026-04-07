# Section 06: Sentiment Fallback Cleanup

## Background

The signal engine collects sentiment data for each symbol via a sentiment collector, then feeds it into synthesis (the `RuleBasedSynthesizer`) and the `SignalBrief` builder. When the sentiment source is unavailable (timeout, API error, no headlines), the collector returns "safe defaults" -- a dict containing `{"sentiment_score": 0.5, "dominant_sentiment": "neutral", "n_headlines": 0, ...}`.

The problem: returning a synthetic 0.5 score is semantically misleading. Although the synthesizer correctly checks `n_headlines > 0` before using the score, other consumers (engine.py's `SignalBrief` builder, `weight_learner.py`) use `.get("sentiment_score", 0.5)` with a fallback, meaning a missing key and a fabricated 0.5 are indistinguishable. Future consumers may treat 0.5 as real signal. Returning `{}` makes the absence of data explicit and forces all consumers to handle it via `.get()` defaults.

## Active Collector Identification

The runtime collector is **`collect_sentiment_alphavantage`** from `src/quantstack/signal_engine/collectors/sentiment_alphavantage.py`. This is confirmed at `src/quantstack/signal_engine/engine.py` line 224:

```python
"sentiment": collect_sentiment_alphavantage(symbol, self._store),
```

The other collector (`collect_sentiment` from `sentiment.py`) is imported at engine.py line 49 but not called in the runtime path. Both collectors must be fixed (the unused one could be activated in the future), but `sentiment_alphavantage.py` is the priority.

## Consumer Audit

All consumers of sentiment collector output and how they handle `{}`:

| Consumer | File | Access Pattern | Safe with `{}`? |
|----------|------|----------------|------------------|
| `RuleBasedSynthesizer._compute_bias_and_conviction` | `synthesis.py` lines 308-319 | `sent = sentiment or {}; sent.get("n_headlines", 0)` then guards on `n_headlines > 0` | Yes -- `.get()` with defaults, guarded |
| `SignalBrief` builder | `engine.py` lines 354-355 | `sentiment.get("sentiment_score", 0.5)` and `sentiment.get("dominant_sentiment", "neutral")` | Yes -- `.get()` provides defaults |
| `weight_learner.py` | `performance/weight_learner.py` lines 440-442 | `sentiment.get("sentiment_score", 0.5)` then guards on `n_headlines > 0` | Yes -- `.get()` with defaults, guarded |

All three consumers use `.get()` with explicit defaults. None will raise `KeyError` on `{}`. No code changes needed in consumers.

## Tests (Write First)

**Test file:** `tests/unit/test_sentiment_fallback.py`

```python
"""Tests for sentiment collector fallback behavior.

Verifies that both sentiment collectors return {} when data is unavailable,
and that the synthesis pipeline handles {} sentiment without error.
"""

# Test: active collector's _safe_defaults returns empty dict
# Import: from quantstack.signal_engine.collectors.sentiment_alphavantage import _safe_defaults
# Assert: _safe_defaults() == {}

# Test: inactive collector's _safe_defaults returns empty dict
# Import: from quantstack.signal_engine.collectors.sentiment import _safe_defaults
# Assert: _safe_defaults() == {}

# Test: active collector returns {} on timeout
# Setup: Patch asyncio.wait_for to raise asyncio.TimeoutError
# Call: await collect_sentiment_alphavantage("TEST", mock_store)
# Assert: result == {}

# Test: active collector returns {} when no headlines available
# Setup: Patch _fetch_alphavantage_headlines to return None
# Call: _collect_sentiment_alphavantage_sync("TEST", mock_store)
# Assert: result == {} (note: the "no_headlines" source tag is gone -- it was part of the old {**_safe_defaults(), "source": "no_headlines"} pattern which now just returns {})

# Test: synthesis handles {} sentiment without error
# Setup: Create a RuleBasedSynthesizer instance
# Call: synthesizer.synthesize(symbol="TEST", technical={...minimal...},
#       regime={"trend_regime": "unknown"}, volume={}, risk={},
#       events={}, fundamentals={}, collector_failures=[],
#       sentiment={})
# Assert: returns a valid SymbolBrief, no KeyError, sentiment contribution is 0.0

# Test: synthesis handles None sentiment without error
# Setup: Same as above but pass sentiment=None
# Assert: returns a valid SymbolBrief, no error
```

## Implementation

### File 1: `src/quantstack/signal_engine/collectors/sentiment_alphavantage.py`

**Change `_safe_defaults()` (line 427-436) to return `{}`:**

Current:
```python
def _safe_defaults() -> dict[str, Any]:
    return {
        "sentiment_score": 0.5,
        "dominant_sentiment": "neutral",
        "n_headlines": 0,
        "reasoning": "",
        "confidence": 0.0,
        "context_used": [],
        "source": "default",
    }
```

New:
```python
def _safe_defaults() -> dict[str, Any]:
    """Returns {} when sentiment data is unavailable.

    Callers must check for empty dict before accessing fields.
    All known consumers use .get() with explicit defaults, so this is safe.
    """
    return {}
```

**Update `_collect_sentiment_alphavantage_sync` (line 74) -- the "no headlines" branch:**

Current:
```python
if not headlines_data:
    return {**_safe_defaults(), "source": "no_headlines"}
```

New:
```python
if not headlines_data:
    return {}
```

The `{**_safe_defaults(), "source": "no_headlines"}` pattern was spreading a full fake-signal dict and overriding one key. With `_safe_defaults()` returning `{}`, this becomes `{"source": "no_headlines"}` which is still misleading (a dict with one key that isn't sentiment data). Return `{}` directly -- the absence of data is the signal.

### File 2: `src/quantstack/signal_engine/collectors/sentiment.py`

**Change `_safe_defaults()` (line 156-162) to return `{}`:**

Current:
```python
def _safe_defaults() -> dict[str, Any]:
    return {
        "sentiment_score": 0.5,
        "dominant_sentiment": "neutral",
        "n_headlines": 0,
        "source": "default",
    }
```

New:
```python
def _safe_defaults() -> dict[str, Any]:
    """Returns {} when sentiment data is unavailable.

    Callers must check for empty dict before accessing fields.
    All known consumers use .get() with explicit defaults, so this is safe.
    """
    return {}
```

**Update `_collect_sentiment_sync` (line 69) -- the "no headlines" branch:**

Current:
```python
if not headlines:
    return {**_safe_defaults(), "source": "no_headlines"}
```

New:
```python
if not headlines:
    return {}
```

Same rationale as the alphavantage collector.

### File 3: `src/quantstack/signal_engine/synthesis.py` -- No Changes Needed

The synthesis code at lines 308-319 already handles `{}` correctly:

```python
sent = sentiment or {}
n_headlines = sent.get("n_headlines", 0)
if n_headlines > 0:
    sent_score = sent.get("sentiment_score", 0.5)
    ...
else:
    scores["sentiment"] = 0.0
```

When `sent` is `{}`, `sent.get("n_headlines", 0)` returns `0`, the guard fails, and `scores["sentiment"]` is set to `0.0`. No sentiment signal leaks through.

### File 4: `src/quantstack/signal_engine/engine.py` -- No Changes Needed

Lines 354-355 use `.get()` with defaults:

```python
sentiment_score=sentiment.get("sentiment_score", 0.5),
dominant_sentiment=sentiment.get("dominant_sentiment", "neutral"),
```

When `sentiment` is `{}`, these resolve to `0.5` and `"neutral"` respectively, which is the same behavior as the old `_safe_defaults()` but now the default lives at the consumer site rather than being fabricated at the producer. This is correct -- the `SignalBrief` schema requires these fields, so the consumer-side default is the right place for them.

## Dependencies

None. This section is fully independent and can be implemented in parallel with all other sections.

## Verification

After implementation:

1. Run `pytest tests/unit/test_sentiment_fallback.py` -- all tests pass
2. Run full test suite to confirm no regressions in synthesis or engine
3. Search codebase for any new consumers of `collect_sentiment` or `collect_sentiment_alphavantage` output that might have been added since this audit and verify they use `.get()` patterns
