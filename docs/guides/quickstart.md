# Quick Start

Get QuantStack running in under 10 minutes.

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | [python.org](https://www.python.org) |
| PostgreSQL | 14+ | `brew install postgresql` |
| tmux | any | `brew install tmux` |
| Claude Code CLI | latest | `npm install -g @anthropic-ai/claude-code` |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

---

## 1. Install

```bash
git clone https://github.com/kbichave/QuantStack.git
cd QuantStack
uv sync --all-extras     # creates .venv and installs all dependencies
```

Or with pip:

```bash
pip install -e ".[all]"
```

---

## 2. Configure

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```bash
TRADER_PG_URL=postgresql://localhost/quantstack   # your PostgreSQL DSN

ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_PAPER=true                                 # paper trading (safe default)

ALPHA_VANTAGE_API_KEY=your_key

USE_REAL_TRADING=false                            # keep false until you're ready
```

Create the database if it doesn't exist:

```bash
createdb quantstack
```

---

## 3. Start

```bash
./start.sh
```

`start.sh` will:
1. Validate all prerequisites and credentials
2. Run DB migrations (idempotent — safe to re-run)
3. Bootstrap the trading universe if empty (~700 symbols)
4. Run preflight checks (kill switch, data freshness, broker connectivity)
5. Display the current credit regime (informational)
6. Compact oversized memory files
7. Launch 4 tmux windows: `trading`, `research`, `supervisor`, `scheduler`

---

## 4. Verify it's running

```bash
# Attach to the tmux session and watch the loops
tmux attach -t quantstack-loops
# Switch windows: Ctrl-b 0 (trading)  Ctrl-b 1 (research)  Ctrl-b 2 (supervisor)
# Detach without stopping: Ctrl-b d
```

After ~5 minutes, verify heartbeats were written:

```bash
python3 -c "
from quantstack.db import open_db
conn = open_db()
rows = conn.execute('''
    SELECT loop_name, MAX(iteration) AS iter, MAX(finished_at) AS last_seen
    FROM loop_heartbeats GROUP BY loop_name
''').fetchall()
for r in rows: print(r)
conn.close()
"
```

Expected output (timestamps ~5 min apart for trading, ~2 min for research):
```
('trading_loop', 1, datetime.datetime(2026, 4, 1, 9, 35, ...))
('research_loop', 1, datetime.datetime(2026, 4, 1, 9, 33, ...))
```

Verify loop state is being persisted (stateless mode working):

```bash
python3 -c "
from quantstack.db import open_db
conn = open_db()
rows = conn.execute('SELECT loop_name, context_key, updated_at FROM loop_iteration_context').fetchall()
for r in rows: print(r)
conn.close()
"
```

---

## 5. Monitor performance

```bash
./report.sh
```

Or query directly:

```bash
python3 -c "
from quantstack.db import open_db
conn = open_db()

# Open positions
positions = conn.execute('SELECT symbol, qty, avg_entry_price FROM positions WHERE status=\'open\'').fetchall()
print('Open positions:', positions)

# Recent fills
fills = conn.execute('''
    SELECT symbol, side, qty, fill_price, realized_pnl, filled_at
    FROM fills ORDER BY filled_at DESC LIMIT 5
''').fetchall()
print('Recent fills:', fills)
conn.close()
"
```

---

## 6. Stop

```bash
tmux kill-session -t quantstack-loops
```

Or activate the kill switch (stops new orders, leaves positions open for manual review):

```bash
python3 -c "
from quantstack.db import open_db
conn = open_db()
conn.execute(\"UPDATE system_state SET value='active', updated_at=NOW() WHERE key='kill_switch'\")
conn.commit()
conn.close()
print('Kill switch activated')
"
```

---

## Troubleshooting

**`start.sh` fails at preflight** — read the error output. Common causes: PostgreSQL not running, `TRADER_PG_URL` wrong, Alpaca keys invalid, `quantstack` package not installed.

**No heartbeats after 10 minutes** — attach to tmux and check the trading window for errors. The most common cause is an expired Alpaca paper account or an invalid Alpha Vantage key.

**`loop_iteration_context` empty** — the loops may be failing on the first tool call. Check `data/logs/trading_loop.log` for Python exceptions.

**Bug-fix watcher not triggering** — check that `record_tool_error()` is being called in the loop (the loop prompts include the pattern). Query the bugs table:

```bash
python3 -c "
from quantstack.db import open_db
conn = open_db()
rows = conn.execute('SELECT tool_name, consecutive_errors, status FROM bugs ORDER BY created_at DESC LIMIT 10').fetchall()
for r in rows: print(r)
conn.close()
"
```

---

## Next steps

- [Architecture overview](../architecture/README.md) — how the components fit together
- [Deployment guide](deployment.md) — environment variables, data paths, utility scripts
- [Execution setup](execution_setup.md) — broker config, risk limits, kill switch
