# Section 03: Transition Zone Propagation

## Objective

Propagate regime transition state from synthesis through to position sizing so that the trading graph can halve position size during uncertain regime transitions. Currently, the transition probability dampens the signal score in synthesis but the downstream position sizer has no visibility into whether a transition is occurring.

## Dependencies

- None for the schema/flag changes (SymbolBrief, feedback_flags)
- **Section 06** wires the synthesis-side setting, but the dataclass and flag changes here are standalone

## Files to Modify

1. **`src/quantstack/shared/schemas.py`** -- add `transition_zone` field to `SymbolBrief`
2. **`src/quantstack/signal_engine/synthesis.py`** -- set `transition_zone=True` when transition probability > 0.3
3. **`src/quantstack/graphs/trading/nodes.py`** -- apply 0.5x position scalar when transition_zone is True
4. **`src/quantstack/config/feedback_flags.py`** -- add `transition_position_sizing_enabled()` flag

## Implementation

### Step 1: Add `transition_zone` to SymbolBrief

File: `src/quantstack/shared/schemas.py`, inside the `SymbolBrief` class (after line 242, the `conviction_factors` field).

```python
    # P05 §5.2: True when regime transition probability > 0.3
    transition_zone: bool = False
```

Place it after `conviction_factors` and before `contributing_pods`. Default is `False` so all existing SymbolBrief construction sites are unaffected.

### Step 2: Set `transition_zone` in synthesis

File: `src/quantstack/signal_engine/synthesis.py`, in the `synthesize()` method.

The existing transition dampening logic is at lines 589-599:

```python
        # --- P05 §5.2: Dampen score during regime transitions ---
        from quantstack.config.feedback_flags import transition_signal_dampening_enabled
        if transition_signal_dampening_enabled():
            transition_prob = regime.get("transition_probability")
            if transition_prob is not None and transition_prob > 0.3:
                ...
```

After computing bias/conviction (but before constructing the SymbolBrief), determine the transition_zone flag:

```python
        # P05 §5.2: Propagate transition zone to brief for downstream sizing
        in_transition_zone = False
        transition_prob = regime.get("transition_probability")
        if transition_prob is not None and transition_prob > 0.3:
            in_transition_zone = True
```

Then in the `SymbolBrief(...)` constructor call (line 368), add:

```python
            transition_zone=in_transition_zone,
```

**Important**: This flag is set regardless of whether `transition_signal_dampening_enabled()` is true. The dampening and the sizing scalar are independent features controlled by separate flags. The transition zone is a factual observation (P(transition) > 0.3), not a behavioral toggle.

### Step 3: Add position sizing flag

File: `src/quantstack/config/feedback_flags.py`, add under the P05 section (after `ensemble_ab_test_enabled`):

```python
def transition_position_sizing_enabled() -> bool:
    """P05 §5.2: Halve position size during regime transitions (transition_zone=True)."""
    return _flag("FEEDBACK_TRANSITION_POSITION_SIZING")
```

Default: False (safe-off), consistent with all other feedback flags.

### Step 4: Apply sizing scalar in trading nodes

File: `src/quantstack/graphs/trading/nodes.py`, in the `make_risk_sizing()` function.

The position sizing scaling happens at lines 740-752, where `signal_value` is multiplied by `affinity` (Wire 2) and `adj` (Wire 4). After these existing scalings, add the transition zone check:

```python
                # Wire P05: Scale down during regime transitions (gated)
                # brief is the SymbolBrief for this candidate's symbol
                try:
                    from quantstack.config.feedback_flags import transition_position_sizing_enabled
                    if transition_position_sizing_enabled():
                        # Look up the SymbolBrief for this symbol from state
                        symbol_briefs = state.get("symbol_briefs", {})
                        brief = symbol_briefs.get(sym)
                        if brief is not None and getattr(brief, "transition_zone", False):
                            signal_value *= 0.5
                            logger.info(
                                "transition_zone_sizing | %s signal_value halved (P(transition) > 0.3)",
                                sym,
                            )
                except Exception:
                    pass  # Non-critical: sizing unchanged on failure
```

**Where to place this**: After the skill_adjustments block (line 752) and before appending to `normalized_candidates` (line 754).

**Note on `symbol_briefs` in state**: The TradingState must carry the SymbolBrief objects. Check how briefs flow through the graph. If they are not currently in TradingState, the code uses `getattr(brief, "transition_zone", False)` as a safe fallback -- when the brief is not available, no scaling is applied. This matches the "safe-off" principle.

## Edge Cases

1. **`transition_probability` missing from regime dict**: `regime.get("transition_probability")` returns None, `in_transition_zone` stays False. No effect.
2. **`transition_probability` is 0.0**: Below 0.3 threshold. No effect.
3. **`transition_probability` is exactly 0.3**: Not > 0.3. No effect. The threshold is exclusive.
4. **SymbolBrief serialization**: `transition_zone` is a `bool` field with default `False`. Pydantic handles serialization/deserialization correctly. No JSON schema issues.
5. **Flag disabled**: When `FEEDBACK_TRANSITION_POSITION_SIZING` is not set (default), the sizing code is a no-op. Signal dampening may still apply independently.
6. **Both dampening AND sizing enabled**: The signal score is halved (dampening) AND the position size is halved (sizing). This is intentional double caution during transitions. If this is too conservative, the operator disables one flag.
7. **SymbolBrief not in TradingState**: The `state.get("symbol_briefs", {}).get(sym)` returns None, `getattr(None, ...)` would fail, but we check `brief is not None` first.

## Tests

File: `tests/unit/signal_engine/test_transition_zone.py`

```python
"""Tests for transition zone propagation from synthesis through sizing."""

def test_symbol_brief_transition_zone_defaults_false():
    """SymbolBrief.transition_zone defaults to False."""
    brief = SymbolBrief(
        symbol="AAPL", market_summary="test",
        consensus_bias="neutral", pod_agreement="mixed",
    )
    assert brief.transition_zone is False

def test_symbol_brief_transition_zone_set_true():
    """SymbolBrief.transition_zone can be set to True."""
    brief = SymbolBrief(
        symbol="AAPL", market_summary="test",
        consensus_bias="neutral", pod_agreement="mixed",
        transition_zone=True,
    )
    assert brief.transition_zone is True

def test_synthesis_sets_transition_zone_when_prob_high():
    """Synthesis sets transition_zone=True when transition_probability > 0.3."""
    # Construct RuleBasedSynthesizer
    # Call synthesize() with regime={"transition_probability": 0.5, ...}
    # Assert: result.transition_zone is True

def test_synthesis_does_not_set_transition_zone_when_prob_low():
    """Synthesis leaves transition_zone=False when transition_probability <= 0.3."""
    # Call synthesize() with regime={"transition_probability": 0.2, ...}
    # Assert: result.transition_zone is False

def test_synthesis_does_not_set_transition_zone_when_prob_missing():
    """Synthesis leaves transition_zone=False when transition_probability is absent."""
    # Call synthesize() with regime={} (no transition_probability key)
    # Assert: result.transition_zone is False

def test_position_sizing_halved_when_transition_zone(monkeypatch):
    """Position sizing applies 0.5x scalar when transition_zone=True and flag enabled."""
    # monkeypatch FEEDBACK_TRANSITION_POSITION_SIZING=true
    # Mock state with symbol_briefs containing a brief with transition_zone=True
    # Assert: signal_value is halved

def test_position_sizing_unchanged_when_flag_disabled(monkeypatch):
    """Position sizing unchanged when flag is disabled (default)."""
    # monkeypatch FEEDBACK_TRANSITION_POSITION_SIZING not set
    # Assert: signal_value unchanged even with transition_zone=True

def test_transition_position_sizing_flag_default():
    """Flag defaults to False."""
    from quantstack.config.feedback_flags import transition_position_sizing_enabled
    assert transition_position_sizing_enabled() is False
```
