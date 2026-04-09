# Section 08: Data Staleness Tiering

## Objective

Replace the flat 8-hour staleness threshold with context-aware tiered thresholds (30 min during market hours, 8 hours during extended hours, 24 hours overnight/weekends). Add auto-disable for dependent signal collectors when data goes stale, and a freshness reporting helper.

## Dependencies

None — can be implemented in parallel with sections 01, 03, 04, 05.

## Files to Create/Modify

### Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/data/validator.py` | Replace `STALE_THRESHOLD_HOURS = 8` with tiered thresholds. Add `get_stale_threshold()` that returns the correct threshold based on current market session. Update `check_freshness()` to use tiered thresholds |
| `src/quantstack/data/validator.py` | Add `get_freshness_report()` helper that returns per-symbol, per-data-type last-updated timestamps |

### New Files (if needed)

| File | Description |
|------|-------------|
| `src/quantstack/data/freshness.py` | (Optional — could live in validator.py) Freshness report query, stale-data event firing, collector auto-disable mapping |

## Implementation Details

### Step 1: Tiered Thresholds

Replace the flat constant in `DataValidator`:

```python
from datetime import timedelta, datetime, time
import pytz

ET = pytz.timezone("US/Eastern")

STALE_THRESHOLDS = {
    "market_hours":   timedelta(minutes=30),   # 09:30-16:00 ET Mon-Fri
    "extended_hours": timedelta(hours=8),       # 04:00-09:30, 16:00-20:00 ET
    "after_hours":    timedelta(hours=24),      # overnight + weekends
}

def get_market_session(now: datetime | None = None) -> str:
    """Determine current market session based on Eastern Time.
    
    Returns one of: 'market_hours', 'extended_hours', 'after_hours'
    """
    if now is None:
        now = datetime.now(ET)
    else:
        now = now.astimezone(ET)
    
    # Weekends are always after_hours
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return "after_hours"
    
    t = now.time()
    if time(9, 30) <= t < time(16, 0):
        return "market_hours"
    elif time(4, 0) <= t < time(9, 30) or time(16, 0) <= t < time(20, 0):
        return "extended_hours"
    else:
        return "after_hours"

def get_stale_threshold(now: datetime | None = None) -> timedelta:
    """Return the staleness threshold for the current market session."""
    session = get_market_session(now)
    return STALE_THRESHOLDS[session]
```

### Step 2: Update check_freshness()

Modify `DataValidator.check_freshness()` to use the tiered threshold:

```python
def check_freshness(self, bars: list[Bar], symbol: str) -> bool:
    """Return True if data is fresh, False if stale."""
    if not bars:
        return False
    
    most_recent = max(bar.timestamp for bar in bars)
    threshold = get_stale_threshold()
    age = datetime.now(pytz.UTC) - most_recent
    
    if age > threshold:
        session = get_market_session()
        logger.warning(
            "[Validator] Stale data: %s last_bar=%s age=%s threshold=%s session=%s",
            symbol, most_recent.isoformat(), age, threshold, session,
        )
        return False
    return True
```

### Step 3: Auto-Disable on Stale Data

When data is stale during market hours, fire an event and optionally disable dependent collectors:

```python
# Data type -> list of signal collectors that depend on it
_COLLECTOR_DEPENDENCIES = {
    "ohlcv_daily": ["momentum", "mean_reversion", "trend"],
    "fundamentals": ["fundamental_value", "earnings_quality"],
    "options_chain": ["implied_vol", "put_call_ratio"],
    # ... map from config or hardcode initially
}

def handle_stale_data(symbol: str, data_type: str) -> None:
    """React to stale data: log, fire event, alert."""
    session = get_market_session()
    
    # Always log
    logger.warning(
        "[Freshness] Stale %s for %s during %s", data_type, symbol, session,
    )
    
    # Insert system alert
    with pg_conn() as conn:
        conn.execute(
            """INSERT INTO system_events
               (event_type, symbol, severity, details, created_at)
               VALUES ('DATA_STALE', %s, %s, %s, NOW())""",
            [symbol, "warning" if session != "market_hours" else "error",
             json.dumps({"data_type": data_type, "session": session})],
        )
    
    # During market hours, disable dependent collectors
    if session == "market_hours":
        for collector in _COLLECTOR_DEPENDENCIES.get(data_type, []):
            logger.warning("[Freshness] Auto-disabling collector: %s", collector)
            # Signal to the collector manager to skip this collector
            # Implementation depends on how collectors are managed
```

### Step 4: Freshness Report

```python
def get_freshness_report() -> dict[str, dict[str, datetime]]:
    """Return {symbol: {data_type: last_updated}} for all universe symbols.
    
    Queries multiple tables to find the most recent timestamp per symbol per data type.
    Used by the supervisor graph for health monitoring.
    """
    report = {}
    
    queries = {
        "ohlcv_daily": "SELECT symbol, MAX(timestamp) as last_updated FROM ohlcv WHERE timeframe = '1d' GROUP BY symbol",
        "fundamentals": "SELECT symbol, MAX(updated_at) as last_updated FROM financial_statements GROUP BY symbol",
        "earnings": "SELECT symbol, MAX(updated_at) as last_updated FROM earnings_calendar GROUP BY symbol",
        "insider_trades": "SELECT symbol, MAX(updated_at) as last_updated FROM insider_trades GROUP BY symbol",
    }
    
    with pg_conn() as conn:
        for data_type, sql in queries.items():
            rows = conn.execute(sql).fetchall()
            for row in rows:
                symbol = row["symbol"]
                if symbol not in report:
                    report[symbol] = {}
                report[symbol][data_type] = row["last_updated"]
    
    return report
```

## Test Requirements

### TDD Tests

```python
# Test: market hours threshold is 30 minutes
def test_market_hours_threshold():
    # Monday at 10:00 ET
    now = datetime(2025, 4, 7, 10, 0, tzinfo=ET)
    threshold = get_stale_threshold(now)
    assert threshold == timedelta(minutes=30)

# Test: extended hours threshold is 8 hours
def test_extended_hours_threshold():
    # Monday at 7:00 ET (pre-market)
    now = datetime(2025, 4, 7, 7, 0, tzinfo=ET)
    threshold = get_stale_threshold(now)
    assert threshold == timedelta(hours=8)

# Test: weekend threshold is 24 hours
def test_weekend_threshold():
    # Saturday at 14:00 ET
    now = datetime(2025, 4, 5, 14, 0, tzinfo=ET)
    threshold = get_stale_threshold(now)
    assert threshold == timedelta(hours=24)

# Test: after hours (overnight) threshold is 24 hours
def test_overnight_threshold():
    # Monday at 2:00 ET
    now = datetime(2025, 4, 7, 2, 0, tzinfo=ET)
    threshold = get_stale_threshold(now)
    assert threshold == timedelta(hours=24)

# Test: stale data fires event and disables collector
def test_stale_data_fires_event(mock_db, monkeypatch):
    monkeypatch.setattr("quantstack.data.validator.get_market_session", lambda: "market_hours")
    handle_stale_data("AAPL", "ohlcv_daily")
    # Verify system_events row was inserted
    # Verify dependent collectors would be disabled

# Test: freshness report returns correct last_updated per symbol
def test_freshness_report(populated_db):
    report = get_freshness_report()
    assert "AAPL" in report
    assert "ohlcv_daily" in report["AAPL"]
    assert isinstance(report["AAPL"]["ohlcv_daily"], datetime)

# Test: get_market_session edge cases
def test_market_session_at_open():
    now = datetime(2025, 4, 7, 9, 30, tzinfo=ET)
    assert get_market_session(now) == "market_hours"

def test_market_session_at_close():
    now = datetime(2025, 4, 7, 16, 0, tzinfo=ET)
    assert get_market_session(now) == "extended_hours"  # 16:00 is extended, not market
```

## Acceptance Criteria

1. `STALE_THRESHOLD_HOURS = 8` is replaced with tiered thresholds (30m / 8h / 24h)
2. `get_market_session()` correctly classifies current time into market_hours, extended_hours, or after_hours
3. `check_freshness()` uses the tiered threshold based on current session
4. Stale data during market hours inserts a `system_events` row with severity `"error"`
5. `get_freshness_report()` returns per-symbol, per-data-type last-updated timestamps
6. Auto-disable mechanism for dependent collectors is implemented (at minimum: logging which collectors would be disabled)
7. All session boundary edge cases are tested (market open, close, pre-market, post-market, weekends)
