#!/usr/bin/env bash
# Monthly performance report.
# Usage: ./report.sh [YYYY-MM]   (defaults to current month)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

MONTH="${1:-$(date +%Y-%m)}"
OUTPUT="reports/monthly_${MONTH}.md"
mkdir -p reports

echo "[report.sh] Generating report for $MONTH → $OUTPUT"

python3 - <<PYEOF
import os, sys
from datetime import datetime, date

try:
    from quantstack.db import open_db
    conn = open_db()
except Exception as e:
    print(f"ERROR: Cannot connect to PostgreSQL: {e}", file=sys.stderr)
    sys.exit(1)

month = "$MONTH"
start = f"{month}-01"
# End = first day of next month
yr, mo = int(month[:4]), int(month[5:7])
if mo == 12:
    end = f"{yr+1}-01-01"
else:
    end = f"{yr}-{mo+1:02d}-01"

# --- Fills ---
fills = conn.execute("""
    SELECT symbol, action, fill_price, quantity, filled_at, strategy_id
    FROM fills
    WHERE filled_at >= %s AND filled_at < %s
    ORDER BY filled_at
""", (start, end)).fetchall()

# --- Daily equity ---
equity_rows = conn.execute("""
    SELECT snapshot_date, total_equity, daily_pnl, daily_return_pct
    FROM daily_equity
    WHERE snapshot_date >= %s AND snapshot_date < %s
    ORDER BY snapshot_date
""", (start, end)).fetchall()

# --- Active strategies ---
strat_rows = conn.execute("""
    SELECT name, status, sharpe_ratio, max_drawdown_pct, win_rate, total_trades
    FROM strategies
    WHERE status IN ('live', 'forward_testing')
    ORDER BY status, sharpe_ratio DESC NULLS LAST
""").fetchall()

conn.close()

# --- Compute metrics ---
total_pnl = sum(r[3] for r in equity_rows) if equity_rows else 0.0
daily_returns = [r[3] for r in equity_rows if r[3] is not None]
starting_equity = equity_rows[0][1] if equity_rows else 0.0
ending_equity = equity_rows[-1][1] if equity_rows else 0.0

import math
sharpe = 0.0
if len(daily_returns) > 1:
    mean_r = sum(daily_returns) / len(daily_returns)
    std_r = math.sqrt(sum((r - mean_r)**2 for r in daily_returns) / (len(daily_returns) - 1))
    sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0

# Max drawdown
max_dd = 0.0
peak = starting_equity
for _, eq, _, _ in equity_rows:
    if eq > peak:
        peak = eq
    dd = (peak - eq) / peak if peak > 0 else 0.0
    if dd > max_dd:
        max_dd = dd

# Win rate from fills (pair buys/sells)
pairs = {}
for sym, action, price, qty, ts, strat in fills:
    if sym not in pairs:
        pairs[sym] = []
    pairs[sym].append((action, price, qty, ts))

wins, losses, total_trades = 0, 0, 0
for sym, trades in pairs.items():
    buys = [(p, q) for a, p, q, _ in trades if a == "buy"]
    sells = [(p, q) for a, p, q, _ in trades if a == "sell"]
    for i, (sp, sq) in enumerate(sells):
        if i < len(buys):
            ep, eq_ = buys[i]
            pnl = (sp - ep) * sq
            if pnl > 0:
                wins += 1
            else:
                losses += 1
            total_trades += 1

win_rate = wins / total_trades if total_trades > 0 else 0.0
total_return_pct = (ending_equity - starting_equity) / starting_equity * 100 if starting_equity > 0 else 0.0

# --- Write report ---
lines = [
    f"# Monthly Performance Report — {month}",
    f"",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"",
    f"## Summary",
    f"",
    f"| Metric | Value |",
    f"|--------|-------|",
    f"| Period | {start} → {end[:-3]} |",
    f"| Starting Equity | \${starting_equity:,.2f} |",
    f"| Ending Equity | \${ending_equity:,.2f} |",
    f"| Total Return | {total_return_pct:+.2f}% |",
    f"| Sharpe (annualized) | {sharpe:.2f} |",
    f"| Max Drawdown | {max_dd*100:.2f}% |",
    f"| Win Rate | {win_rate*100:.1f}% ({wins}W / {losses}L / {total_trades} trades) |",
    f"| Trading Days | {len(equity_rows)} |",
    f"",
    f"## Active Strategies",
    f"",
    f"| Strategy | Status | Sharpe | Max DD | Win Rate | Trades |",
    f"|----------|--------|--------|--------|----------|--------|",
]
for name, status, sr, md, wr, tt in strat_rows:
    lines.append(
        f"| {name} | {status} | {sr:.2f if sr else 'n/a'} | "
        f"{md:.1f if md else 'n/a'}% | {wr*100:.1f if wr else 'n/a'}% | {tt or 0} |"
    )

lines += [
    f"",
    f"## Daily P&L",
    f"",
    f"| Date | Equity | Daily P&L | Daily Return |",
    f"|------|--------|-----------|--------------|",
]
for snap_date, eq, pnl, ret in equity_rows[-20:]:  # last 20 days
    lines.append(f"| {snap_date} | \${eq:,.2f} | \${pnl:+,.2f} | {ret:+.2f}% |")

with open("$OUTPUT", "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Report written to $OUTPUT")
print(f"  Return: {total_return_pct:+.2f}%  Sharpe: {sharpe:.2f}  MaxDD: {max_dd*100:.1f}%  Trades: {total_trades}")
PYEOF
