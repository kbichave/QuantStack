# Section 09: Conflicting Signal Resolution

## Overview

When the signal engine's collectors disagree strongly -- technical indicators say bullish, ML says bearish, sentiment is neutral -- the weighted average produces a middling conviction that masks the actual uncertainty. A conviction of 0.55 could mean "all collectors weakly agree" or "half are screaming buy and half are screaming sell." These are fundamentally different risk profiles, but today the system treats them identically.

This section adds conflict detection to the `RuleBasedSynthesizer` in `src/quantstack/signal_engine/synthesis.py`. When the spread between the highest and lowest collector vote scores exceeds 0.5, the system flags the signal as conflicting, caps conviction at 0.3, and publishes a `SIGNAL_CONFLICT` event via the EventBus for downstream consumption by the research graph and daily planner.

## Dependencies

- **section-06-eventbus-extension**: Must be completed first. This section publishes `SIGNAL_CONFLICT` events, which requires the `EventType.SIGNAL_CONFLICT` enum member added in section-06.

## Blocks

- None. No downstream sections depend on this one.

## Background: Current Synthesis Logic

The `RuleBasedSynthesizer._compute_bias_and_conviction()` method in `src/quantstack/signal_engine/synthesis.py` works as follows:

1. Each collector produces a vote score in [-1.0, +1.0]: trend, rsi, macd, bb, sentiment, ml, flow.
2. Votes are multiplied by regime-conditional weights (from `_WEIGHT_PROFILES`) and summed into a single `score`.
3. Conviction is computed as `abs(score)`, then adjusted additively (+0.10 for strong ADX, -0.15 for weekly contradicting daily, etc.).
4. Final conviction is clamped to [0.05, 0.95].

The problem: step 2 averages out disagreement. A +1.0 and -1.0 cancel to 0.0, producing low conviction that looks like "no signal" rather than "high conflict." There is no mechanism to distinguish between "collectors have nothing to say" and "collectors violently disagree."

## Tests First

File: `tests/unit/test_conflict_resolution.py`

### Test: conflict detected when spread exceeds threshold

Given a `scores` dict where the maximum vote is +0.8 (e.g., trend) and the minimum vote is +0.2 (e.g., ml), the spread is 0.6 which exceeds the 0.5 threshold. The conflict detection function should return `True`.

More critically: when one collector votes +1.0 and another votes -1.0 (spread = 2.0), conflict must be detected.

### Test: no conflict when spread is within threshold

Given a `scores` dict where the maximum vote is +0.6 and the minimum vote is +0.2 (spread = 0.4), no conflict is detected. The function should return `False`.

### Test: boundary -- spread exactly 0.5 is not conflicting

The detection rule is strictly greater than 0.5 (`max - min > 0.5`). A spread of exactly 0.5 should not trigger conflict detection.

### Test: conviction capped at 0.3 when signals conflict

When conflict is detected, the final conviction value must not exceed 0.3, regardless of the base conviction computed from the weighted average. If the base conviction is 0.7 and conflict is detected, the output conviction should be 0.3.

### Test: conviction not capped when signals do not conflict

When no conflict is detected, the conviction cap does not apply. A base conviction of 0.7 with no conflict should remain 0.7 (subject to other adjustments and the existing [0.05, 0.95] clamp).

### Test: SIGNAL_CONFLICT event published with correct payload

When conflict is detected, a `SIGNAL_CONFLICT` event is published via the EventBus with a payload containing:
- `symbol`: the symbol being synthesized
- `conflicting_collectors`: list of collector names involved (at minimum the max and min voters)
- `max_signal`: the highest vote score
- `min_signal`: the lowest vote score
- `spread`: the computed spread value

Mock the EventBus and verify `publish()` is called once with these fields.

### Test: no event published when no conflict

When the spread is below the threshold, verify that no `SIGNAL_CONFLICT` event is published. The EventBus `publish()` should not be called for this event type.

### Test: conflict detection uses raw vote scores, not weighted scores

The spread check must operate on the raw per-collector vote scores (the `scores` dict values in [-1.0, +1.0]), not on the weight-multiplied values. A collector with a small weight but an extreme vote still represents genuine disagreement in the signal landscape.

### Test: zero-score collectors are included in spread calculation

A collector voting 0.0 (neutral) should be included when computing max and min. If one collector votes +0.8 and another votes 0.0, the spread is 0.8 and conflict is detected.

## Implementation

### File to modify

`src/quantstack/signal_engine/synthesis.py`

### Change 1: Add conflict detection logic to `_compute_bias_and_conviction`

Inside the `_compute_bias_and_conviction` method, after the `scores` dict is fully populated (all 7 collectors have voted) and before the conviction clamping step, add conflict detection:

1. Compute the spread: `spread = max(scores.values()) - min(scores.values())`
2. If `spread > 0.5`, flag the signal as conflicting.
3. When conflicting, cap conviction at 0.3 (configurable constant `CONFLICT_CONVICTION_CAP = 0.3`).
4. The cap applies after all other conviction adjustments (ADX, stability, weekly contradiction, etc.) but before the final `max(0.05, min(0.95, conviction))` clamp.

The conflict check should be placed in the method at the point after the conviction adjustments and before the final clamp (around the current line 420 in synthesis.py). The logic:

```python
# --- Conflict detection ---
vote_values = [v for v in scores.values() if v is not None]
signal_spread = max(vote_values) - min(vote_values)
is_conflicting = signal_spread > 0.5

if is_conflicting:
    conviction = min(conviction, 0.3)
```

The constant `0.5` for the spread threshold and `0.3` for the conviction cap should be defined as class-level constants on `RuleBasedSynthesizer`:

```python
CONFLICT_SPREAD_THRESHOLD = 0.5
CONFLICT_CONVICTION_CAP = 0.3
```

### Change 2: Add EventBus publishing

The `_compute_bias_and_conviction` method currently has no access to the symbol name or EventBus. Two options:

**Option A (preferred):** Move the conflict detection and event publishing into the `synthesize()` method, which already has access to `symbol`. After calling `_compute_bias_and_conviction()`, check for conflict and publish. This keeps `_compute_bias_and_conviction` pure (returns data, no side effects) and puts the EventBus call at the orchestration layer.

Concretely, in `synthesize()`:

1. After `bias, conviction = self._compute_bias_and_conviction(...)`, also get the scores and conflict state. This requires `_compute_bias_and_conviction` to return the `scores` dict alongside `bias` and `conviction` (change return type to a 3-tuple or a small dataclass).

2. If conflicting, publish:
```python
event_bus.publish(
    EventType.SIGNAL_CONFLICT,
    payload={
        "symbol": symbol,
        "conflicting_collectors": _identify_conflicting_collectors(scores),
        "max_signal": max(scores.values()),
        "min_signal": min(scores.values()),
        "spread": signal_spread,
    },
)
```

**Option B (simpler):** Accept an optional `event_bus` parameter on `synthesize()` and pass `symbol` into `_compute_bias_and_conviction`. This avoids changing the return type but mixes side effects into the computation method.

The choice depends on how much refactoring is acceptable. Option A is cleaner but touches the return type contract. Option B is less invasive.

For EventBus access: the `RuleBasedSynthesizer` does not currently hold a reference to the EventBus. Either inject it via the constructor (`__init__(self, event_bus=None)`) or accept it as an optional parameter on `synthesize()`. Constructor injection is preferable since the synthesizer is instantiated once and reused.

### Change 3: Identify conflicting collectors for the event payload

Add a helper to identify which collectors are on opposite sides of the conflict:

```python
def _identify_conflicting_collectors(scores: dict[str, float]) -> list[str]:
    """Return collector names that are on the extreme ends of the vote spread."""
    max_score = max(scores.values())
    min_score = min(scores.values())
    max_collectors = [k for k, v in scores.items() if v == max_score]
    min_collectors = [k for k, v in scores.items() if v == min_score]
    return max_collectors + min_collectors
```

### Change 4: Structured logging for conflict events

Log every conflict detection for pattern analysis. Use structured logging via loguru:

```python
logger.warning(
    "Signal conflict detected",
    symbol=symbol,
    spread=signal_spread,
    scores=scores,
)
```

Over time, consistent conflict between specific collector pairs (e.g., technical RSI always conflicting with ML direction) may indicate that one should be removed or that the conflict itself is predictive.

### Constants summary

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `CONFLICT_SPREAD_THRESHOLD` | `0.5` | `RuleBasedSynthesizer` class | Minimum spread to trigger conflict detection |
| `CONFLICT_CONVICTION_CAP` | `0.3` | `RuleBasedSynthesizer` class | Maximum conviction when conflict detected |

### Integration with section-10 (conviction multiplicative)

Section-10 converts the additive conviction adjustments to multiplicative factors. The conflict conviction cap must apply **after** the multiplicative computation but **before** the final clamp. The ordering is:

1. Compute base conviction from weighted vote average
2. Apply multiplicative factors (section-10, if enabled)
3. Apply conflict cap (this section): `if conflicting: conviction = min(conviction, 0.3)`
4. Final clamp to [0.05, 0.95]

This ordering ensures the conflict cap is authoritative -- no amount of favorable multiplicative factors can override genuine collector disagreement.

### Import additions

```python
from quantstack.coordination.event_bus import EventBus, EventType
```

This import is only needed if EventBus publishing is added directly in this file. If the caller (signal engine) handles publishing externally, no import changes are needed in synthesis.py.

## Rollback

Remove the conflict detection check from `_compute_bias_and_conviction` or `synthesize()`. Conviction reverts to the existing behavior where disagreeing collectors simply cancel out in the weighted average. Any `SIGNAL_CONFLICT` events already in the `loop_events` table remain as inert TEXT rows. No data loss, no functional impact on other components.
