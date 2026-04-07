# 05 — Graph Restructuring: The System Can't Complete a Trading Cycle

**Priority:** P1-P2 — Architectural Blocker
**Timeline:** Week 3-6
**Gate:** Trading graph completes in <300s. Governance separated from execution. Missing agents deployed.

---

## Why This Section Exists

This is the finding that changes the architecture conversation. **All 3 graphs are in chronic timeout.** The trading graph's critical path is 1,910s (31.8 minutes) — 6.3x its 300s cycle interval. The 600s watchdog kills it every time. After 3 consecutive kills, the system halts. This means the trading graph as designed **cannot actually run a complete cycle during market hours.**

This isn't a bug to fix with timeout tuning. It's a design that can't work. The graphs must be broken apart and rebuilt.

---

## The Timeout Crisis

### All 3 Graphs Are in Chronic Timeout

| Graph | Cycle Interval | Watchdog | Realistic Runtime | Status |
|-------|---------------|----------|-------------------|--------|
| **Trading** | 300s (5 min) | 600s | **1,910s (31.8 min)** | Every cycle killed at 600s. 3 failures → halt. |
| **Research** | 120s | 600s | 766-1,666s | Times out on feedback loops |
| **Supervisor** | 300s | 600s | 685-1,085s | Times out on community intel weeks |

### Trading Graph Critical Path: Why 31.8 Minutes

```
data_refresh           30s  ─┐
safety_check            5s   │
market_intel          300s   │  SEQUENTIAL
plan_day              300s   │  (must complete in order)
                             │
  ┌─ entry_scan       600s ──┤  PARALLEL BRANCH 1
  └─ position_review  480s   │  (max = 600s)
                             │
risk_sizing             5s   │
portfolio_construction 10s   │
                             │
  ┌─ portfolio_review 300s ──┤  PARALLEL BRANCH 2
  └─ analyze_options  300s   │  (max = 300s)
                             │
execute_entries        60s   │
reflect               300s ──┘

TOTAL: 1,910s worst case
```

The bottleneck isn't parallelism — the two branch points already save ~780s. The bottleneck is **too many heavy LLM calls in series**: `market_intel`(300s) → `plan_day`(300s) → `entry_scan`(600s) → `portfolio_review`(300s) → `reflect`(300s). That's 1,800s of just LLM reasoning on the critical path.

---

## The Agent Problem: Wrong Agents, Not Enough Agents

We have 21 agents. 6 of them are ghosts (learning modules exist but never called). And we're missing at least 6 critical roles a real fund staffs.

### What We Have That's Redundant or Overloaded

| Agent | Problem |
|-------|---------|
| **trade_debater** | 600s timeout, 22 tools, heavy Sonnet. Plays both bull AND bear. One agent doing two adversarial jobs = neither done well. |
| **fund_manager** | Conflates governance (allocation), execution (sizing), and compliance (concentration check). Three jobs, one LLM call. |
| **position_monitor + exit_evaluator** | Two agents in series (300s + 180s) for one task: "should we exit?" Could be one agent with clearer scope. |
| **trade_reflector** | Writes quality scores that nobody reads. Its output is decorative until the feedback loop is wired. |

### What We're Missing Entirely

| Missing Role | Why It Matters | Who Does This at a Fund |
|-------------|---------------|------------------------|
| **Broker Reconciler** | Position drift, phantom fills, margin calls | Operations desk |
| **Compliance Monitor** | Wash sales, PDT, restricted securities | Compliance officer |
| **Greeks Monitor** | Portfolio delta/gamma/theta/vega exposure | Risk desk |
| **Data Quality Monitor** | Stale OHLCV, outliers, bad indicators | Data engineering |
| **Factor Exposure Monitor** | Sector tilt, momentum crowding, correlation shifts | Portfolio analytics |
| **Corporate Actions Handler** | Splits, dividends, M&A → adjust positions | Operations desk |

**Note:** Broker reconciliation was initially listed as missing in the Quant Scientist audit but was found to exist in `guardrails/agent_hardening.py:463-550` and `execution_monitor._reconcile_loop()`. The reconciliation *function* exists — what's missing is a dedicated *monitoring agent* that runs it on schedule and escalates issues.

---

## The Solution: 5-Graph Architecture

The current 3-graph design is wrong. Here's what it should be:

### Graph 1: Market Intelligence (Event-Driven, Not Cyclic)

**Runs:** Once at 8:30 AM + on >2% index move events
**Agents:** `market_intel`, `intraday_regime_detector` (new)
**Cycle:** Event-triggered, not 5-min polling
**Output:** `market_context` cached in DB, consumed by other graphs

**Why separate:** `market_intel` runs every 5-min cycle today but only produces meaningful output pre-market or on big moves. The other 80% of cycles, it's a 300s no-op that burns tokens and **blocks the trading critical path**.

### Graph 2: Research Pipeline (Async, Long-Running)

**Runs:** 10-min cycles during market hours, overnight autoresearch mode
**Agents:** `quant_researcher`, `ml_scientist`, `strategy_rd`, `hypothesis_critic`, `domain_researcher`
**Cycle:** 600-1800s (let it run to completion, not kill at 600s)
**Output:** Draft strategies → `strategies` table, knowledge graph updates

**Change from today:** Increase watchdog to 1,800s. Research shouldn't be on a tight clock. Let it think. The current 120s cycle interval with 600s watchdog means research is constantly being killed mid-hypothesis.

### Graph 3: Trading Decisions (5-Min Cycle, TIGHT)

**Runs:** Every 5 min during market hours
**Agents:** `daily_planner` (once/day), `entry_scanner` (Haiku, not Sonnet), `exit_evaluator`, `compliance_monitor` (new)
**Cycle:** Must complete in <300s
**Output:** Entry/exit signals → `pending_orders` table

**The key change:** Strip this graph to ONLY decision-making. No execution. No reflection. No portfolio optimization. Just: *"Given current signals and positions, what should we buy/sell?"*

```
data_refresh (15s, deterministic)
  → safety_check (5s, deterministic — no LLM)
  → exit_evaluation (60s, Haiku — per-position SL/TP/thesis check)
  → entry_scan (120s, Haiku — not 600s Sonnet)
  → compliance_check (10s, deterministic — wash sale, PDT)
  → write pending_orders to DB

TOTAL: ~210s — fits in 300s cycle comfortably
```

**Why this works:** The 600s `trade_debater` on Sonnet is THE bottleneck. Replace it with a 120s Haiku scanner that applies rules from the daily mandate (set by Graph 4). The deep thinking happens once/day in governance, not every 5 minutes.

### Graph 4: Governance & Portfolio (Once/Day + On-Demand)

**Runs:** Once at 9:00 AM, again at 3:30 PM, + on significant portfolio events
**Agents:** `CIO` (new, heavy Sonnet), `fund_manager` (portfolio construction only), `options_analyst`, `risk_officer` (new, deterministic + Haiku)
**Cycle:** 30-60 min allowed (this is where deep thinking belongs)
**Output:** Daily mandate → `daily_mandate` table

**Morning session (9:00 AM):**
1. CIO analyzes regime + signals + overnight research
2. Sets daily mandate: "momentum gets 40% of capital, mean_rev 30%, options 30%"
3. Sets position limits: "max 3 new entries, max $5K per position"
4. Sets restrictions: "no TSLA today (earnings tomorrow), no naked options"
5. `fund_manager` runs portfolio optimization: target weights
6. `options_analyst` pre-screens vol structures worth monitoring
7. `risk_officer` validates: factor exposure, sector concentration, Greeks budget

**Afternoon session (3:30 PM):**
1. Review day's performance
2. Adjust mandate for last 30 min if needed
3. Pre-set overnight position parameters

**Why separate:** Governance thinking is expensive but infrequent. The CIO agent uses Sonnet with extended thinking, runs for 10-20 minutes, and produces a mandate that constrains the fast Trading Decision graph for the rest of the day. This is the OrgAgent pattern (AR-4) — **74% token reduction** because expensive reasoning happens 2x/day instead of 80x/day.

### Graph 5: Supervision & Operations (Always-On)

**Runs:** Always, 5-min health checks, daily batch jobs
**Agents:** `health_monitor`, `self_healer`, `strategy_promoter`, `broker_reconciler` (new), `data_quality_monitor` (new), `greeks_monitor` (new)
**Cycle:** 5 min for health, daily for batch
**Output:** Events to EventBus, alerts, auto-recovery actions

**Changes from today:**
- Add `broker_reconciler`: every 5 min, compare DB positions vs broker positions. Alert on drift.
- Add `data_quality_monitor`: every 15 min, check data freshness, outliers, missing symbols.
- Add `greeks_monitor`: every 5 min for options positions, compute portfolio Greeks, alert on limit breach.
- `strategy_promoter` and `community_intel` move to daily/weekly batch (not every 5-min cycle).

---

## The Agent Count After Restructuring

| Graph | Agents | Tier | Cycle | Cost/Day |
|-------|--------|------|-------|----------|
| 1. Market Intel | 2 (`market_intel`, `regime_detector`) | Medium | Event-driven | ~$2 |
| 2. Research | 5 (`quant_researcher`, `ml_scientist`, `strategy_rd`, `critic`, `domain`) | Heavy + Medium | 10-30 min | ~$15 |
| 3. Trading Decisions | 4 (`planner_daily`, `entry_scanner`, `exit_evaluator`, `compliance`) | Medium + Deterministic | 5 min | ~$5 |
| 4. Governance | 4 (`CIO`, `fund_manager`, `options_analyst`, `risk_officer`) | Heavy | 2x/day | ~$3 |
| 5. Supervision | 6 (`health`, `healer`, `promoter`, `reconciler`, `data_quality`, `greeks`) | Medium + Light | 5 min | ~$5 |
| **TOTAL** | **21** (same count, completely different distribution) | | | **~$30/day** |

**vs. today's estimated $150-$450/day** (Sonnet on every 5-min cycle).

---

## The Three Architectural Truths

### 1. The trading graph can never complete.

1,910s runtime in a 600s watchdog is not a bug to fix — it's a design that can't work. Must be split.

### 2. Expensive thinking should happen rarely.

CIO + deep research = 2-3 heavy calls/day. Fast execution = 80+ cheap calls/day. Today it's backwards — Sonnet runs 80x/day in the trading loop.

### 3. Missing agents matter more than existing agents.

No broker reconciler means we can't trust position state. No compliance monitor means legal risk. No Greeks monitor means options blow up silently. Adding these 3 agents matters more than optimizing the 21 we have.

---

## Migration Path: 3 Graphs → 5 Graphs

This is not a big-bang rewrite. It's a strangler-fig migration:

### Phase A: Extract Market Intel (Week 3-4, 3 days)

| Step | Action |
|------|--------|
| 1 | Create Graph 1 service in `docker-compose.yml` |
| 2 | Move `market_intel` node out of trading graph → Graph 1 |
| 3 | Graph 1 writes `market_context` to DB |
| 4 | Trading graph reads `market_context` from DB (replaces inline call) |
| 5 | **Immediate savings:** 300s removed from trading critical path |

**Risk:** Low — market_intel already produces a JSON context that gets passed to downstream nodes. Moving it to a separate service and writing to DB is a clean interface change.

### Phase B: Extract Governance (Week 4-5, 5 days)

| Step | Action |
|------|--------|
| 1 | Create Graph 4 service |
| 2 | Create `daily_mandate` table schema |
| 3 | Split `fund_manager` into CIO (governance) + fund_manager (sizing only) |
| 4 | Split `trade_debater` into `entry_scanner` (Haiku, fast, rule-based within mandate) |
| 5 | Graph 3 (Trading) reads `daily_mandate` at cycle start, constrains decisions |
| 6 | **Immediate savings:** 600s debater + 300s plan_day removed from 5-min cycle |

**Risk:** Medium — this changes the decision flow. The CIO's daily mandate must be comprehensive enough that the fast trading loop doesn't need to "think deeply." Requires careful mandate schema design.

### Phase C: Add Missing Agents to Supervision (Week 5-6, 4 days)

| Step | Action |
|------|--------|
| 1 | Add `broker_reconciler` node to supervisor graph |
| 2 | Add `data_quality_monitor` node |
| 3 | Add `greeks_monitor` node |
| 4 | Add `compliance_monitor` node to trading graph (deterministic, no LLM) |
| 5 | Move `strategy_promoter` to daily batch schedule |

**Risk:** Low — these are additive. They don't change existing flows, they add new monitoring.

### Phase D: Increase Research Watchdog (Week 3, 1 hour)

| Step | Action |
|------|--------|
| 1 | Change research graph watchdog from 600s to 1,800s |
| 2 | Change research cycle interval from 120s to 600s |
| 3 | Let research run to completion instead of being killed |

**Risk:** Very low — config change only.

---

## Impact Summary

| Metric | Before (3 graphs) | After (5 graphs) | Improvement |
|--------|-------------------|-------------------|-------------|
| Trading cycle time | 1,910s (always killed at 600s) | ~210s (fits in 300s) | **9x faster** |
| Trading cycle completion rate | ~0% (always times out) | ~95%+ | **Actually works** |
| LLM cost/day | $150-$450 | ~$30 | **80-93% reduction** |
| Missing critical agents | 6 | 0 | All roles staffed |
| Governance frequency | 80x/day (every 5-min Sonnet cycle) | 2x/day (morning + afternoon) | Appropriate cadence |
| Research freedom | Killed at 600s mid-hypothesis | 1,800s to complete | Research can think |
| Position safety | No reconciliation, no compliance, no Greeks monitoring | All three active | Operational safety |

---

## Summary: Graph Restructuring Delivery

| Phase | Item | Effort | Impact |
|-------|------|--------|--------|
| A | Extract Market Intel to Graph 1 | 3 days | -300s from trading path |
| B | Extract Governance to Graph 4 | 5 days | -900s from trading path, 80% cost reduction |
| C | Add missing agents to Supervision | 4 days | Compliance, reconciliation, Greeks monitoring |
| D | Increase research watchdog | 1 hour | Research completes instead of being killed |

**Total estimated effort: 12-13 engineering days.**
**Annual cost savings: $44,000-$153,000 (LLM tokens alone).**
**Operational impact: Trading graph actually completes cycles for the first time.**
