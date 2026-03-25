# Skills

Claude Code slash commands for QuantPod. These are for **human-invoked workflows only**.
Autonomous trading and research is handled by the loops in `prompts/`.

---

## User-Invocable Skills

| Skill | Command | When to Use |
|-------|---------|-------------|
| [lit-review](lit_review.md) | `/lit-review` | Academic research → product gap analysis |
| [get-alerts](get-alerts.md) | `/get-alerts` | View and manage equity/investment alerts |
| [update-data](update-data.md) | `/update-data` | Audit DB coverage and fill data gaps |
| [compact-memory](compact_memory.md) | `/compact-memory` | Distil memory files when any exceeds 200 lines |

---

## When to Use Each

```
Research:            /lit-review
Data maintenance:    /update-data
Alert management:    /get-alerts
Memory hygiene:      /compact-memory  (run when memory files > 200 lines)
Reflection:          automatic — trade-reflector agent fires on close (per_trade) and every 10th close / Friday EOD (weekly_review)
```

---

## Agents (spawned by loops — not user-invocable)

Trading loop agents live in `.claude/agents/`:

| Agent | Spawned By | Role |
|-------|-----------|------|
| `market-intel` | trading_loop Step 1d | Real-time news and macro intel |
| `position-monitor` | trading_loop Step 2 | Per-position exit recommendations |
| `trade-debater` | trading_loop Steps 2+3 | Bull/bear/risk debate |
| `fund-manager` | trading_loop Step 3f | Batch portfolio approval |
| `options-analyst` | trading_loop Step 3g | Options structure selection and validation |
| `earnings-analyst` | trading_loop Step 3c | Earnings event structure selection |
| `trade-reflector` | trading_loop Step 2f (on close) | Root cause classification + lesson extraction |
| `risk` | trading_loop (deep analysis) | VaR, Kelly sizing, stress testing |
| `quant-researcher` | research_loop | Multi-week hypothesis generation |
| `ml-scientist` | research_loop | ML training and feature quality |
| `strategy-rd` | research_loop | Overfitting gates, strategy lifecycle |
| `execution-researcher` | research_loop (monthly) | TCA, fill quality, factor exposure audit |
