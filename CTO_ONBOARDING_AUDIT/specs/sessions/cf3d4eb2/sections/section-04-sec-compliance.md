# Section 04: SEC Compliance (PDT, Wash Sale, Tax Lots, Margin)

## Overview

This section implements four regulatory compliance domains in a new `src/quantstack/execution/compliance/` package. The account is below $25K, making PDT enforcement the critical path — every trade without it is a FINRA 4210 violation risk.

**Depends on:** section-02-fill-legs (fill_legs table and dual-write for post-trade hooks), section-03-business-calendar (business day arithmetic for PDT 5-day window, wash sale 30-day window)

**Blocks:** section-14-integration-tests

**Enforcement model summary:**

| Domain | Enforcement | Integration Point |
|--------|-------------|-------------------|
| PDT (FINRA 4210) | Hard block pre-trade | Risk gate — reject order |
| Wash Sale (IRC 1091) | Post-trade accounting + pre-trade warning | Fill hooks + risk gate warning |
| Tax Lots (Form 8949) | Post-trade accounting | Fill hooks (FIFO matching) |
| Margin (Reg T) | Pre-trade check | Risk gate — reject order |

---

## Tests

All tests go in `tests/unit/execution/compliance/`. The test file structure mirrors the module structure.

### PDT Checker Tests (`tests/unit/execution/compliance/test_pretrade.py`)

```python
# --- PDTChecker ---

# Test: 0 day trades in 5-day window -> order APPROVED
# Test: 2 day trades in 5-day window -> order APPROVED
# Test: 3 day trades in 5-day window AND account < $25K -> 4th day trade REJECTED
# Test: 3 day trades in 5-day window AND account >= $25K -> order APPROVED
# Test: day trade on Monday, window counts only business days (skip weekend)
# Test: day trade counting resets after 5 business days roll forward
# Test: partial fill that closes intraday position counts as day trade
# Test: options PDT matches on full OCC contract symbol, not underlying
# Test: two different SPY option contracts closed same day count as 2 day trades
# Test: position opened yesterday and closed today is NOT a day trade
```

### Day Trade Recording Tests (`tests/unit/execution/compliance/test_pretrade.py`)

```python
# --- Day Trade Recording ---

# Test: buy then sell same symbol same day creates day_trade record
# Test: buy then sell different symbol same day does NOT create day_trade record
# Test: multiple round-trips same symbol same day creates multiple records
```

### Margin Calculator Tests (`tests/unit/execution/compliance/test_pretrade.py`)

```python
# --- MarginCalculator ---

# Test: long equity order requires 50% cash margin
# Test: long option order requires premium as margin
# Test: debit spread requires net premium as margin
# Test: order rejected when margin_required exceeds available equity
# Test: reducing position does NOT require additional margin
```

### Wash Sale Tracker Tests (`tests/unit/execution/compliance/test_posttrade.py`)

```python
# --- WashSaleTracker ---

# Test: sell at loss creates pending_wash_losses record with 30-day window
# Test: sell at gain does NOT create pending_wash_losses record
# Test: buy within 30 days of pending loss -> wash sale flagged, loss disallowed
# Test: buy after 30 days of pending loss -> no wash sale
# Test: wash sale adjusts cost basis of replacement shares by disallowed amount
# Test: pending_wash_losses marked resolved after buy triggers
# Test: pre-trade warning surfaces for buy with open wash window
# Test: pre-trade warning does NOT block the order
```

### Tax Lot Manager Tests (`tests/unit/execution/compliance/test_posttrade.py`)

```python
# --- TaxLotManager ---

# Test: buy fill creates tax lot with correct cost basis and date
# Test: sell fill matches FIFO -- oldest lot consumed first
# Test: sell of 150 shares with lots [100@$50, 100@$55] -> first lot fully consumed, second partially
# Test: gain/loss computed correctly per lot
# Test: wash sale adjustment added to cost basis before gain/loss calculation
# Test: selling more shares than open lots raises error (or handles gracefully)
```

---

## Database Schema

Four new tables, all additive (no modifications to existing tables).

### `day_trades` Table

Tracks every day trade for PDT counting. A day trade is opening AND closing the same symbol on the same trading day.

```python
@dataclass
class DayTrade:
    id: int                  # auto-increment PK
    symbol: str              # full OCC symbol for options, ticker for equity
    open_order_id: str       # FK to orders
    close_order_id: str      # FK to orders
    trade_date: date         # business day the round-trip occurred
    quantity: int            # shares/contracts involved
    account_equity: float    # equity snapshot at time of close
```

Index on `trade_date` for the rolling 5-business-day window query.

### `pending_wash_losses` Table

Holds realized losses that could become wash sales if the same symbol is repurchased within 30 calendar days.

```python
@dataclass
class PendingWashLoss:
    id: int                        # auto-increment PK
    symbol: str
    loss_amount: float             # the realized loss (negative number)
    sell_order_id: str             # FK to orders
    sell_date: date
    window_end: date               # sell_date + 30 calendar days
    resolved: bool                 # True when matched by a subsequent buy
    resolved_by_order_id: str | None
```

Index on `(symbol, window_end)` for efficient lookups on buy events.

### `wash_sale_flags` Table

Records confirmed wash sale events linking the loss trade to the replacement purchase.

```python
@dataclass
class WashSaleFlag:
    id: int                        # auto-increment PK
    loss_trade_id: int             # FK to closed_trades
    replacement_order_id: str      # the buy that triggered wash sale
    disallowed_loss: float         # loss amount that cannot be claimed
    adjusted_cost_basis: float     # new cost basis on replacement shares
    wash_window_start: date
    wash_window_end: date
    flagged_at: datetime
```

### `tax_lots` Table

Tracks individual purchase lots for FIFO matching on sells and Form 8949 reporting.

```python
@dataclass
class TaxLot:
    lot_id: int                    # auto-increment PK
    symbol: str
    quantity: int                  # remaining open quantity
    original_quantity: int         # quantity at creation
    cost_basis: float              # per share, adjusted for wash sales
    acquired_date: date
    order_id: str                  # the buy order that created this lot
    closed_date: date | None
    exit_price: float | None
    realized_pnl: float | None
    wash_sale_adjustment: float    # amount added to cost basis (default 0.0)
    status: str                    # "open" | "closed"
```

Index on `(symbol, status)` for FIFO matching queries (open lots ordered by `acquired_date ASC`).

---

## Module Structure

Create `src/quantstack/execution/compliance/` as a new package:

```
src/quantstack/execution/compliance/
    __init__.py          # exports PDTChecker, MarginCalculator, WashSaleTracker, TaxLotManager
    pretrade.py          # PDTChecker, MarginCalculator
    posttrade.py         # WashSaleTracker, TaxLotManager
```

The business day calendar is provided by section-03 at `src/quantstack/execution/compliance/calendar.py`. This section consumes it but does not implement it.

---

## Pre-Trade Components (`pretrade.py`)

### PDTChecker

Callable from the risk gate. Determines whether a proposed order would trigger a 4th day trade in a rolling 5-business-day window for an account below $25K.

**`PDTChecker.check(order, account_equity, positions) -> ComplianceResult`**

Logic:
1. Determine if this order would close a position opened on the same business day (use business calendar from section-03 for "same day" determination).
2. If not a potential day trade, return APPROVED immediately.
3. Query `day_trades` table for records where `trade_date` falls within the rolling 5-business-day window ending today. Use business calendar to compute the window start (5 business days back, skipping weekends and market holidays).
4. If count >= 3 AND `account_equity` < 25000.0, return REJECTED with reason `"PDT: 4th day trade blocked (account < $25K)"`.
5. Otherwise return APPROVED.

**Day trade detection details:**
- A "same symbol" match uses the exact symbol string. For options, this is the full OCC contract symbol (e.g., `SPY240119C00450000`), not the underlying ticker. Two different SPY option contracts are NOT the same position for PDT purposes.
- Multiple round-trips on the same symbol same day each count as a separate day trade.
- A partial fill that brings the position to zero (closing a same-day open) counts.

**Post-trade recording (also in pretrade.py or a shared helper):**
After every fill that closes a position, check if that position was opened on the same business day. If so, insert a `day_trades` record. This recording feeds the next pre-trade check.

### MarginCalculator

Callable from the risk gate. Checks whether the proposed order would exceed available margin.

**`MarginCalculator.check(order, positions, account_equity) -> ComplianceResult`**

Logic:
1. Compute current margin used across all open positions:
   - Long equity: `position_notional * 0.50` (Reg T 50% initial margin)
   - Long options: premium paid (max loss = premium)
   - Debit spreads: net premium paid
2. Compute margin required for the proposed order using the same rules.
3. If `current_margin_used + proposed_margin > account_equity`, return REJECTED with reason `"Margin: insufficient equity (required: $X, available: $Y)"`.
4. Sell/close orders do NOT require additional margin (they reduce exposure).

**Scope note:** Credit spread and naked option margin calculations are deferred until those strategies are traded. Initially support long equity, long options, and debit spreads only.

---

## Post-Trade Components (`posttrade.py`)

### WashSaleTracker

Called after every fill via the fill hook system. Implements two-phase wash sale detection.

**`WashSaleTracker.on_fill(fill_event) -> WashSaleResult`**

**Phase 1 — On every realized-loss sell:**
1. Determine if the fill realizes a loss (compare fill price to cost basis from tax lots).
2. If loss: query whether the same symbol was bought within 30 calendar days BEFORE this sell date. If yes, the loss is an immediate wash sale (retroactive detection from the look-back window).
3. Insert a `pending_wash_losses` record with `window_end = sell_date + 30 calendar days` regardless. This flags the loss as potentially washable by a future buy.

**Phase 2 — On every buy:**
1. Query `pending_wash_losses` for the same symbol where `window_end >= today` AND `resolved = false`.
2. If found: this buy triggers wash sale treatment on the pending loss.
3. Create a `wash_sale_flags` record linking the loss trade to this replacement purchase.
4. Adjust the cost basis of the new shares (tax lot) by adding the disallowed loss amount: `new_cost_basis = purchase_price + (disallowed_loss / quantity)`.
5. Mark the `pending_wash_losses` entry as `resolved = true`, set `resolved_by_order_id`.

**Pre-trade warning (surfaced via risk gate, NOT a block):**
When processing a buy order in the risk gate, check `pending_wash_losses` for the same symbol with `window_end >= today` and `resolved = false`. If found, attach a warning to the risk gate result: `"Wash sale warning: buying {symbol} within 30-day window of realized loss on {sell_date}"`. The order proceeds — this is informational only.

### TaxLotManager

Called after every fill via the fill hook system. Maintains FIFO lot tracking.

**`TaxLotManager.on_fill(fill_event) -> TaxLotResult`**

**On every buy fill:**
1. Create a new `tax_lots` record: `status="open"`, `quantity=fill_qty`, `cost_basis=fill_price`, `acquired_date=fill_date`, `wash_sale_adjustment=0.0`.
2. If a wash sale applies (from `WashSaleTracker`), adjust: `cost_basis += disallowed_loss / quantity`, `wash_sale_adjustment = disallowed_loss`.

**On every sell fill:**
1. Query open tax lots for the symbol, ordered by `acquired_date ASC` (FIFO).
2. Walk lots oldest-first, consuming shares:
   - If lot has enough shares: reduce `quantity` by sell amount, compute `realized_pnl = (sell_price - cost_basis) * matched_qty`.
   - If lot is fully consumed: set `status="closed"`, `closed_date=today`, `exit_price=sell_price`.
   - If lot is partially consumed: reduce `quantity`, keep `status="open"`.
3. Continue until all sell shares are matched.
4. If sell quantity exceeds total open lots, this is an error condition (short selling without lots). Log an error and handle gracefully — do not crash the fill pipeline.

---

## Risk Gate Integration

Two new checks are inserted into the risk gate's pre-trade check sequence in `src/quantstack/execution/risk_gate.py`.

**Insertion point:** After the existing daily loss limit check (check #4 in the current layered sequence) and before the existing liquidity/volume checks.

### PDT Check

```python
# In risk_gate.py pre-trade check sequence:
# ... existing checks 1-4 ...

pdt_result = PDTChecker.check(order, account_equity, positions)
if pdt_result.rejected:
    return RiskGateResult(approved=False, reason=pdt_result.reason)

# ... continue to margin check ...
```

This is a HARD BLOCK. If the PDT checker rejects, the order does not proceed. No override, no flag-only mode. The account is below $25K and a 4th day trade is a regulatory violation.

### Margin Check

```python
margin_result = MarginCalculator.check(order, positions, account_equity)
if margin_result.rejected:
    return RiskGateResult(approved=False, reason=margin_result.reason)

# ... continue to existing volume/liquidity checks ...
```

Also a HARD BLOCK. Insufficient margin means the broker would reject anyway — catching it in the risk gate provides a better error message and avoids wasted API calls.

### Wash Sale Warning

```python
# For buy orders only:
wash_warning = WashSaleTracker.check_pending(order.symbol)
if wash_warning:
    result.warnings.append(wash_warning)
    # Do NOT block — warning only
```

The warning is surfaced in the risk gate result, logged in the audit trail, and visible in trade logs. It does not prevent the order.

---

## Fill Hook Integration

After every fill is recorded (using the fill hook system from section-02), two post-trade compliance checks run:

```python
# In the fill processing pipeline (order_lifecycle.py or hook_registry.py):

# 1. Check for day trade (PDT recording)
PDTChecker.record_if_day_trade(fill_event, positions)

# 2. Tax lot management (must run before wash sale so lots exist)
TaxLotManager.on_fill(fill_event)

# 3. Wash sale detection (reads tax lots, may adjust cost basis)
WashSaleTracker.on_fill(fill_event)
```

The ordering matters: tax lots must be created/consumed before wash sale logic runs, because wash sale adjustments modify cost basis on the newly created lot.

---

## Edge Cases and Design Decisions

### PDT Edge Cases
- **Options day trades:** Match on the full OCC contract symbol. `SPY240119C00450000` and `SPY240119P00450000` are different instruments — closing one does not create a day trade for the other.
- **Multiple round-trips:** If you buy AAPL, sell AAPL, buy AAPL, sell AAPL all in one day, that is 2 day trades.
- **Partial fills closing intraday positions:** If the fill brings the position quantity to zero and the position was opened today, it counts.
- **Multi-leg option spreads:** Each leg is an independent trade for PDT purposes (conservative interpretation per FINRA guidance).

### Wash Sale Edge Cases
- **Look-forward problem:** Solved by two-phase detection. At time of loss sale, we cannot know if the same symbol will be repurchased within 30 days. The `pending_wash_losses` table holds the loss, and subsequent buys check against it.
- **Multiple pending losses for same symbol:** Each loss gets its own `pending_wash_losses` record. A single buy can resolve multiple pending losses if there are overlapping windows.
- **Substantially identical securities:** For initial implementation, "substantially identical" is defined as exact symbol match. This is conservative for equities (same ticker) but does NOT catch options on the same underlying as substantially identical to the stock. This is a known simplification — full IRS interpretation is complex and deferred.

### Margin Edge Cases
- **Options with no clear max loss:** Only long options and debit spreads are in scope. Their max loss is bounded (premium paid or net debit). Credit spreads and naked options are deferred.
- **Margin during partial fills:** Margin is checked at order submission time against the full order size, not per partial fill.

---

## Implementation Checklist

1. Create `src/quantstack/execution/compliance/__init__.py` — export public classes.
2. Create `src/quantstack/execution/compliance/pretrade.py` — `PDTChecker`, `MarginCalculator`, `ComplianceResult` dataclass.
3. Create `src/quantstack/execution/compliance/posttrade.py` — `WashSaleTracker`, `TaxLotManager`, result dataclasses.
4. Add schema migration for `day_trades`, `pending_wash_losses`, `wash_sale_flags`, `tax_lots` tables.
5. Integrate `PDTChecker.check()` into `risk_gate.py` pre-trade sequence (after daily loss limit, before volume checks).
6. Integrate `MarginCalculator.check()` into `risk_gate.py` pre-trade sequence (after PDT check).
7. Integrate `WashSaleTracker.check_pending()` as a warning in the risk gate for buy orders.
8. Wire `PDTChecker.record_if_day_trade()`, `TaxLotManager.on_fill()`, and `WashSaleTracker.on_fill()` into the fill hook pipeline.
9. Write all tests in `tests/unit/execution/compliance/test_pretrade.py` and `test_posttrade.py`.
10. Verify PDT hard block works end-to-end: 3 day trades + account < $25K + attempted 4th close = REJECTED.
