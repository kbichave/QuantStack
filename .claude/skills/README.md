# Skills

Claude Code slash commands for the QuantPod trading system. Type `/skill-name` to invoke.

---

## User-Invocable Skills

| Skill | Command | When to Use | Frequency |
|-------|---------|-------------|-----------|
| [trade](trade.md) | `/trade` | Run TradingCrew analysis for a symbol and decide whether to enter, hold, or skip | Daily, before market open |
| [review](review.md) | `/review` | Check open positions, P&L, stop proximity, and strategy health | Daily or mid-session |
| [meta](meta.md) | `/meta` | Orchestrate across multiple symbols — allocate, resolve conflicts, batch trade | When managing >1 position or symbol |
| [workshop](workshop.md) | `/workshop` | Research and backtest a new strategy hypothesis | When a regime slot is empty or a strategy has failed |
| [reflect](reflect.md) | `/reflect` | Review recent outcomes, update memory, fix broken skills and IC prompts | Weekly or after 10+ trades |
| [decode](decode.md) | `/decode` | Reverse-engineer a strategy from external signal history or your own trade log | When you see a pattern worth formalising |
| [tune](tune.md) | `/tune` | Edit IC and pod manager prompts based on accuracy data | After 3+ reflect sessions or when an IC accuracy < 50% |
| [compact-memory](compact_memory.md) | `/compact-memory` | Distil memory files, remove stale/redundant entries | When any memory file exceeds 200 lines, or after 5+ sessions |

---

## Reference (Internal — Not User-Invocable)

| File | Purpose |
|------|---------|
| [deep_analysis.md](deep_analysis.md) | QuantCore tool guide — when to call which raw-data, options, and risk tools to enrich a DailyBrief analysis |

---

## Session Flow

Typical day:
```
/review  →  /meta  →  /trade
```

Weekly (Friday close):
```
/reflect  →  (if IC accuracy issues) /tune
```

Strategy R&D (any time):
```
/workshop  →  (if decoding external signals) /decode
```

---

## Automated Triggers (scheduler.py)

| Time (ET) | Days | Trigger |
|-----------|------|---------|
| 09:15 | Mon–Fri | `/review → /meta → /trade` |
| 12:30 | Mon–Fri | `/review` |
| 15:45 | Mon–Fri | `/review` |
| 17:00 | Friday | `/reflect` |

Run manually: `python scripts/scheduler.py --run-now morning_routine`

---

## Workshop Prompt History

Versioned prompts live in `tmp/` (gitignored). Current series:

| File | Status | Description |
|------|--------|-------------|
| `tmp/workshop_v4_quality_rsimr.md` | Done — forward_testing | Quality/low-beta filter (XOM, IBM, MSFT) |
| `tmp/workshop_v5_mtf_rsimr.md` | **Next** | Multi-timeframe: Swing/Medium/Day-trade profiles on XOM, IBM, MSFT |
