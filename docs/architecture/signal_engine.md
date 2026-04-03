# Signal Engine

## Purpose

The Signal Engine is QuantStack's primary market analysis pipeline. It replaces the
original 13-agent CrewAI pod (TradingCrew) with a pure-Python, zero-LLM implementation
that runs in 2-6 seconds wall-clock time. The engine orchestrates 16 concurrent data
collectors, synthesizes their outputs through a rule-based system with regime-conditional
weights, and produces a structured `SignalBrief` consumed by every downstream system
(Trading Graph entry scanning, AutonomousRunner, position monitoring, and `/reflect`).

Every call produces a valid `SignalBrief`. There are no failure modes that return `None`
or raise to the caller -- partial or total collector failure degrades gracefully to a
neutral brief with zero confidence.

---

## Architecture

```
                          SignalEngine.run(symbol)
                                  |
                   asyncio.gather (10s per-collector timeout)
            ┌──────┬──────┬──────┼──────┬──────┬──────┬──────┐
            v      v      v      v      v      v      v      v
        technical regime volume risk events fundamentals ... social
            |      |      |      |      |      |      |      |
            └──────┴──────┴──────┼──────┴──────┴──────┴──────┘
                                 v
                      RuleBasedSynthesizer.synthesize()
                      map_to_market_bias()
                      map_to_risk_environment()
                                 |
                                 v
                           SignalBrief
                                 |
                      DriftDetector.check_drift_from_brief()
                      (best-effort, never blocks)
```

1. **Collector phase** -- All 16 collectors run concurrently via `asyncio.gather`. Each
   gets a 10-second wall-clock timeout (`_COLLECTOR_TIMEOUT`). A failed collector
   returns `{}` and its name is appended to `collector_failures`.

2. **Synthesis phase** -- `RuleBasedSynthesizer` computes a `SymbolBrief` from the
   collected signals. `map_to_market_bias()` and `map_to_risk_environment()` aggregate
   symbol-level signals into brief-level fields. Regime-conditional weight profiles
   ensure that indicator weighting adapts to market conditions (e.g., RSI gets 25%
   weight in ranging markets but only 10% in trending-up markets).

3. **Drift detection** -- `DriftDetector` runs a PSI (Population Stability Index) check
   against the brief. This is best-effort and never blocks brief delivery.

4. **Output** -- A fully populated `SignalBrief` is returned. Confidence is penalized
   by 0.05 per failed collector, floored at 0.1.

---

## Collector Catalog

16 collectors are registered in the engine. Two additional modules exist on disk
(`sentiment.py` -- the original pre-v1.1 collector, and `l2_microstructure.py`) but
are not wired into the current pipeline.

| # | Name | Category | Module | Description |
|---|------|----------|--------|-------------|
| 1 | `technical` | Price/Technical | `technical.py` | RSI, MACD, Bollinger Bands, ADX, trend indicators |
| 2 | `regime` | Macro/Regime | `regime.py` | Market regime detection (trending_up/down, ranging) |
| 3 | `volume` | Price/Technical | `volume.py` | Volume ratio, VWAP, accumulation/distribution |
| 4 | `risk` | Risk | `risk.py` | VIX level, ATR, drawdown, portfolio-level risk |
| 5 | `events` | Events | `events.py` | Earnings dates, ex-dividend, economic calendar |
| 6 | `fundamentals` | Fundamentals | `fundamentals.py` | EPS, P/E, revenue growth, balance sheet metrics |
| 7 | `sentiment` | Sentiment | `sentiment_alphavantage.py` | News sentiment via Alpha Vantage + Groq reasoning (replaced original `sentiment.py` in v1.1) |
| 8 | `macro` | Macro/Regime | `macro.py` | Yield curve slope, rate regime (rising/falling/stable) |
| 9 | `sector` | Macro/Regime | `sector.py` | Sector relative strength, rotation signal, breadth |
| 10 | `flow` | Flow | `flow.py` | Institutional flow signal, insider buying/selling |
| 11 | `cross_asset` | Macro/Regime | `cross_asset.py` | Risk-on/off regime from cross-asset signals |
| 12 | `quality` | Fundamentals | `quality.py` | Earnings quality composite score |
| 13 | `ml_signal` | Quant | `ml_signal.py` | Trained ML model predictions (probability + direction) |
| 14 | `statarb` | Quant | `statarb.py` | Pairs/spread signals, z-score of spread |
| 15 | `options_flow` | Flow | `options_flow_collector.py` | Dealer positioning: GEX, gamma flip, DEX, max pain, IV skew, VRP, charm, vanna, EHD |
| 16 | `social` | Sentiment | `social_sentiment.py` | Reddit + Stocktwits community sentiment |

All collector modules live in `src/quantstack/signal_engine/collectors/`.

Each collector is an async function with the signature:

```python
async def collect_<name>(symbol: str, store: DataStore) -> dict[str, Any]:
```

A collector that fails or times out returns `{}`. The engine never retries a failed
collector within a single run -- retries happen at the next scheduled invocation.

---

## SignalBrief Schema

`SignalBrief` is a Pydantic `BaseModel` defined in `brief.py`. It is a strict superset
of `DailyBrief` -- every field present in `DailyBrief` exists with an identical name,
type, and semantic. This invariant is enforced by `tests/test_signal_brief_schema.py`:

```python
DailyBrief.model_validate(signal_brief.model_dump())  # must not raise
```

### Core fields (DailyBrief-compatible)

| Field | Type | Description |
|-------|------|-------------|
| `date` | `date` | Analysis date |
| `market_overview` | `str` | Human-readable market summary |
| `market_bias` | `bullish / bearish / neutral` | Aggregated directional bias |
| `market_conviction` | `float [0, 1]` | Strength of the bias signal |
| `risk_environment` | `low / normal / elevated / high` | Current risk regime |
| `symbol_briefs` | `list[SymbolBrief]` | Per-symbol analysis detail |
| `top_opportunities` | `list[str]` | Symbols meeting bullish + high-conviction threshold |
| `key_risks` | `list[str]` | Top 3 risk factors from the symbol brief |
| `overall_confidence` | `float [0, 1]` | Pipeline confidence, penalized by failures |
| `pods_reporting` | `int` | Count of successful collectors |
| `total_analyses` | `int` | Count of collector outputs received |

### SignalEngine-specific additions

| Field | Type | Default | Source collector |
|-------|------|---------|-----------------|
| `engine_version` | `str` | `"signal_engine_v1"` | -- |
| `collection_duration_ms` | `float` | `0.0` | -- |
| `collector_failures` | `list[str]` | `[]` | -- |
| `regime_detail` | `dict or None` | `None` | regime |
| `sentiment_score` | `float [0, 1]` | `0.5` | sentiment |
| `dominant_sentiment` | `str` | `"neutral"` | sentiment |
| `macro_rate_regime` | `str` | `"unknown"` | macro |
| `yield_curve_slope` | `float or None` | `None` | macro |
| `sector_signal` | `str` | `"unknown"` | sector |
| `rotation_signal` | `str` | `"unknown"` | sector |
| `flow_signal` | `float or None` | `None` | flow |
| `insider_direction` | `str` | `"unknown"` | flow |
| `cross_asset_regime` | `str` | `"unknown"` | cross_asset |
| `risk_on_score` | `float or None` | `None` | cross_asset |
| `quality_score` | `float or None` | `None` | quality |
| `ml_prediction` | `float or None` | `None` | ml_signal |
| `ml_direction` | `str` | `"unknown"` | ml_signal |
| `statarb_signal` | `str` | `"unknown"` | statarb |
| `spread_zscore` | `float or None` | `None` | statarb |
| `opt_gex` | `float or None` | `None` | options_flow |
| `opt_gamma_flip` | `float or None` | `None` | options_flow |
| `opt_dex` | `float or None` | `None` | options_flow |
| `opt_max_pain` | `float or None` | `None` | options_flow |
| `opt_iv_skew` | `float or None` | `None` | options_flow |
| `opt_vrp` | `float or None` | `None` | options_flow |
| `drift_warning` | `bool` | `False` | drift_detector |
| `social` | `dict` | `{}` | social |

The `to_daily_brief()` method returns a `DailyBrief` by dropping the extra fields.

---

## Synthesis

`RuleBasedSynthesizer` (in `synthesis.py`) converts raw collector outputs into a
`SymbolBrief`. It is deterministic -- no LLM calls, no randomness.

### Regime-conditional weights

The synthesizer selects a weight profile based on the detected regime. Weights are
applied to six signal voters: `trend`, `rsi`, `macd`, `bb`, `sentiment`, `ml`, and
`flow`. All weights sum to 1.0.

| Regime | trend | rsi | macd | bb | sentiment | ml | flow |
|--------|-------|-----|------|----|-----------|----|------|
| `trending_up` | 0.35 | 0.10 | 0.20 | 0.05 | 0.10 | 0.15 | 0.05 |
| `trending_down` | 0.30 | 0.15 | 0.20 | 0.05 | 0.10 | 0.15 | 0.05 |
| `ranging` | 0.05 | 0.25 | 0.10 | 0.25 | 0.10 | 0.15 | 0.10 |

When the ML collector does not return a valid prediction, its weight is redistributed
proportionally across the other voters.

Design rationale: RSI oversold in a trending-down market is a falling knife, not a buy
signal. MACD momentum in a ranging market is noise, not signal. Regime-conditional
weights encode these priors.

### Aggregation functions

- `map_to_market_bias(symbol_briefs)` -- aggregates per-symbol consensus bias into a
  brief-level `bullish / bearish / neutral` with conviction.
- `map_to_risk_environment(symbol_briefs, regime_outputs)` -- aggregates risk factors
  and regime data into `low / normal / elevated / high`.

---

## Fault Tolerance

The engine is designed so that no single collector failure can prevent brief delivery.

1. **Per-collector timeout** -- Each collector gets a 10-second `asyncio.wait_for`
   timeout. A slow external API (Alpha Vantage, Groq, Alpaca) cannot stall the pipeline.

2. **Exception isolation** -- `asyncio.gather(*coros, return_exceptions=True)` catches
   all exceptions. Failed collectors produce `{}` and are recorded in
   `collector_failures`.

3. **Confidence penalty** -- Each failed collector reduces `overall_confidence` by 0.05,
   floored at 0.1. Downstream consumers (entry scanning, position sizing) use this to
   reduce exposure when data is incomplete.

4. **All-failures fallback** -- If every collector fails, the engine returns a brief
   with `market_bias="neutral"`, `overall_confidence=0.0`, `analysis_quality="low"`.
   This is a valid `SignalBrief` that passes schema validation.

5. **Multi-symbol isolation** -- `run_multi()` wraps each symbol in a try/except. A
   failure for one symbol produces an empty brief for that symbol; other symbols are
   unaffected.

---

## Drift Detection

After synthesis, `DriftDetector.check_drift_from_brief()` runs a PSI check on tracked
features extracted from the brief. This is best-effort -- it never blocks brief delivery,
and exceptions are caught and logged at DEBUG level.

### Tracked features

`rsi_14`, `atr_pct`, `adx_14`, `bb_pct`, `volume_ratio`, `regime_confidence`

### PSI thresholds

| PSI range | Severity | Action |
|-----------|----------|--------|
| < 0.10 | NONE | Distributions stable, no action |
| 0.10 - 0.25 | WARNING | Moderate shift, reduce position size |
| >= 0.25 | CRITICAL | Significant shift, skip symbol entries |

### CRITICAL drift response

When PSI >= 0.25 (CRITICAL):

1. `brief.drift_warning` is set to `True`. The Step 3b quality gate skips entries for
   symbols with `drift_warning=True`.
2. An `ml_arch_search` task is inserted into `research_queue` (priority 8) with the
   drifted features and PSI value as context. AutoResearchClaw picks this up and
   investigates whether the current model needs retraining or feature engineering.

### Implementation details

- Pure numpy computation, < 1ms per check.
- Baselines are stored as JSON in `~/.quantstack/drift_baselines/`.
- Missing baselines return severity NONE (no baseline = nothing to drift from).
- No database writes in the hot path; only reads + numpy math. The `research_queue`
  insert on CRITICAL is outside the hot path and wrapped in its own try/except.

---

## Multi-Symbol Execution

`run_multi(symbols, max_concurrent=5)` runs the full pipeline for multiple symbols with
bounded concurrency via `asyncio.Semaphore`. Each symbol gets its own independent
`run()` call. A failure for one symbol does not affect others -- it returns an empty
brief with `collector_failures=["all"]`.

```python
briefs = await SignalEngine().run_multi(["XOM", "MSFT", "SPY"])
```

---

## Key Files

| File | Role |
|------|------|
| `src/quantstack/signal_engine/engine.py` | Orchestrator: collector dispatch, timeout handling, brief assembly |
| `src/quantstack/signal_engine/brief.py` | `SignalBrief` Pydantic model (DailyBrief superset) |
| `src/quantstack/signal_engine/synthesis.py` | `RuleBasedSynthesizer`, regime-conditional weights, aggregation |
| `src/quantstack/signal_engine/collectors/*.py` | 16 active collectors + 2 legacy/unused modules |
| `src/quantstack/learning/drift_detector.py` | PSI-based drift detection, baseline management |
| `src/quantstack/shared/schemas.py` | `DailyBrief`, `SymbolBrief`, `KeyLevel` shared types |
| `tests/test_signal_brief_schema.py` | Backward-compat invariant: `SignalBrief` validates as `DailyBrief` |
