# Section 01: Congressional Trades Collector

## Objective

Build a signal collector that ingests congressional trading disclosures (via Quiver Quantitative API) and produces a directional signal score. Congressional trades have documented alpha (IC 0.03-0.06) due to information asymmetry — legislators trade on non-public policy knowledge with a 5-30 day lead time.

## Files to Create

### `src/quantstack/signal_engine/collectors/congressional.py`

New collector module following the established `async collect_*(symbol, store) -> dict` pattern.

**Implementation details:**

1. **API client** — HTTP client for Quiver Quantitative free-tier API (`https://api.quiverquant.com/beta/live/congresstrading`). Requires `QUIVER_API_KEY` env var. Use `httpx.AsyncClient` with a 8-second timeout.

2. **Signal computation** — `compute_congressional_signal(transactions: list[dict]) -> dict[str, Any]`:
   - Filter to trades within the last 90 days (filings can be delayed up to 45 days).
   - Separate buys (transaction_type "Purchase") from sells ("Sale", "Sale (Full)", "Sale (Partial)").
   - Compute `net_buy_count = len(buys) - len(sells)`.
   - Compute `net_buy_value = sum(buy amounts) - sum(sell amounts)`. Quiver provides amount ranges (e.g., "$1,001 - $15,000"); use midpoint of range.
   - Compute `congress_signal_score` in [-1.0, 1.0]: `tanh(net_buy_count / 5) * 0.6 + tanh(net_buy_value / 500_000) * 0.4`.
   - Detect `bipartisan_buy`: buys from both parties → stronger signal (bump score by 0.1, capped at 1.0).
   - Detect `committee_insider`: members on relevant committees (Finance, Commerce, Energy) get 1.5x weight.
   - Return dict with keys: `congress_signal_score`, `net_buy_count`, `net_buy_value`, `bipartisan`, `committee_insider_count`, `trade_count`, `confidence`.

3. **Entry point** — `async def collect_congressional(symbol: str, store: DataStore) -> dict[str, Any]`:
   - Call `check_freshness(symbol, "congressional_trades", max_days=7)` — skip if stale data flag is recent enough.
   - Wrap API call in `asyncio.wait_for(..., timeout=8.0)`.
   - On any exception, log warning and return `{}`.

4. **Amount range parser** — `parse_amount_range(amount_str: str) -> float`: converts Quiver's textual ranges to numeric midpoints. Handle edge cases: "$1,001 - $15,000" → 8000.5, ">$50,000,000" → 50_000_000.

5. **Committee mapping** — Dict mapping committee names to relevant sector tickers, used to boost signal when a committee member trades in their oversight sector.

## Files to Modify

### `src/quantstack/signal_engine/staleness.py`

Add entry to `STALENESS_THRESHOLDS`:
```python
"congressional_trades": 7,
```

## Test Requirements

### `tests/unit/signal_engine/test_congressional_collector.py`

1. **test_compute_signal_net_buy** — 5 buys, 1 sell → positive score.
2. **test_compute_signal_net_sell** — 1 buy, 4 sells → negative score.
3. **test_compute_signal_empty** — No transactions → score 0.0, confidence 0.0.
4. **test_bipartisan_boost** — Buys from both "D" and "R" → score boosted.
5. **test_committee_insider_weight** — Finance committee member trading bank stock → higher weight.
6. **test_parse_amount_range** — All Quiver amount formats parsed correctly.
7. **test_parse_amount_range_edge_cases** — ">$50,000,000", "$1,001 - $15,000", malformed strings.
8. **test_collect_api_failure** — Mock httpx raising timeout → returns `{}`.
9. **test_collect_missing_api_key** — No `QUIVER_API_KEY` → returns `{}` with warning log.
10. **test_staleness_skip** — When freshness check fails, collector returns `{}` without API call.

## Acceptance Criteria

- [ ] `collect_congressional(symbol, store)` returns a dict with `congress_signal_score` in [-1.0, 1.0] or `{}` on failure.
- [ ] No exceptions propagate — all errors caught and logged.
- [ ] API key read from env var; missing key returns `{}` gracefully.
- [ ] Amount range parser handles all documented Quiver formats.
- [ ] Signal computation is a pure function (no I/O), independently testable.
- [ ] All 10 unit tests pass.
- [ ] Staleness threshold registered at 7 days.
