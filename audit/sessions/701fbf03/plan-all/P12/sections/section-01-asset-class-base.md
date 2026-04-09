# Section 01: Asset Class Base Framework

## Objective

Define the abstract base class and supporting types that all asset classes (futures, crypto, forex, fixed income) implement. This is the foundation every other P12 section depends on.

## Files to Create

### `src/quantstack/asset_classes/__init__.py`
Package init. Re-exports `AssetClass`, `AssetClassType`, `PositionLimits`, `TradingSchedule`.

### `src/quantstack/asset_classes/base.py`
Abstract base class defining the contract every asset class must satisfy:

```python
class AssetClass(ABC):
    def get_data_providers(self) -> list[DataProvider]: ...
    def get_risk_model(self) -> RiskModel: ...
    def get_signal_collectors(self) -> list[Collector]: ...
    def get_execution_adapter(self) -> BrokerAdapter: ...
    def get_trading_hours(self) -> TradingSchedule: ...
    def get_position_limits(self) -> PositionLimits: ...
```

### `src/quantstack/asset_classes/types.py`
Supporting dataclasses and enums:

- **`AssetClassType(Enum)`** — `EQUITY`, `OPTIONS`, `FUTURES`, `CRYPTO`, `FOREX`, `FIXED_INCOME`
- **`TradingSchedule`** — `open_time`, `close_time`, `timezone`, `days_active: list[int]`, `is_24h: bool`. Method `is_open(dt: datetime) -> bool`.
- **`PositionLimits`** — `max_pct_equity: float`, `max_notional: float`, `max_positions: int`, `max_leverage: float`
- **`DataProvider(ABC)`** — abstract with `fetch_ohlcv(symbol, start, end)` and `fetch_quote(symbol)`
- **`RiskModel(ABC)`** — abstract with `margin_requirement(symbol, qty, price)` and `validate_order(order)`
- **`BrokerAdapter(ABC)`** — abstract with `submit_order(order)`, `cancel_order(order_id)`, `get_positions()`

### `src/quantstack/asset_classes/registry.py`
Singleton registry mapping `AssetClassType` to `AssetClass` instances:

- `register(asset_class_type, instance)` — register an asset class
- `get(asset_class_type) -> AssetClass` — retrieve by type
- `enabled() -> list[AssetClassType]` — return all enabled asset classes
- Load enabled state from `asset_class_config` DB table (section-06)

## Files to Modify

### `src/quantstack/config/feedback_flags.py`
Add `MULTI_ASSET_ENABLED: bool = False` feature flag to gate the new code paths.

## Implementation Details

1. All abstract methods raise `NotImplementedError` with clear messages.
2. `TradingSchedule.is_open()` must handle timezone-aware datetimes and DST transitions correctly. Use `zoneinfo.ZoneInfo` (stdlib, no new deps).
3. `PositionLimits` should be frozen dataclasses to prevent mutation after construction.
4. The registry should be thread-safe (use `threading.Lock`) since graph nodes may register concurrently during startup.
5. `BrokerAdapter` should extend the existing adapter pattern in `src/quantstack/execution/adapters/` rather than introducing a parallel hierarchy. Check if the existing `adapters/__init__.py` already defines a base — if so, inherit from it.

## Test Requirements

### `tests/unit/asset_classes/test_base.py`
- Verify that instantiating `AssetClass` directly raises `TypeError` (abstract).
- Create a minimal concrete subclass; verify all methods are callable.
- Verify `TradingSchedule.is_open()` for in-hours, out-of-hours, weekend, and DST boundary.
- Verify `PositionLimits` is immutable (frozen dataclass).

### `tests/unit/asset_classes/test_registry.py`
- Register and retrieve an asset class.
- `get()` for unregistered type raises `KeyError`.
- `enabled()` returns only registered types.
- Thread-safety: concurrent registrations don't corrupt state.

## Acceptance Criteria

- [ ] `AssetClass` ABC is importable from `quantstack.asset_classes`
- [ ] All six abstract methods defined with type annotations
- [ ] `TradingSchedule.is_open()` handles timezone + DST correctly
- [ ] `PositionLimits` is frozen/immutable
- [ ] Registry is thread-safe and tested
- [ ] Feature flag `MULTI_ASSET_ENABLED` added
- [ ] All tests pass under `uv run pytest tests/unit/asset_classes/`
