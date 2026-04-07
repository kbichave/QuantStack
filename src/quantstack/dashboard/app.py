"""Real-time agent dashboard — per-team chat windows.

Lightweight FastAPI app that streams agent events via SSE and serves
a single-page HTML dashboard showing what each team is thinking/doing.

Run standalone:
    uvicorn quantstack.dashboard.app:app --host 0.0.0.0 --port 8421

Or via Docker:
    docker compose up dashboard
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="QuantStack Agent Dashboard", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

# Agent name → team fallback (for events that predate the backend fix)
_AGENT_TEAM_FALLBACK = {
    "quant_researcher": "research", "ml_scientist": "research",
    "position_monitor": "trading", "entry_scanner": "trading",
    "exit_manager": "trading", "risk_assessor": "trading",
    "daily_planner": "trading", "fund_manager": "trading",
    "options_analyst": "trading", "trade_debater": "trading",
    "reflector": "trading", "execution_manager": "trading",
    "health_monitor": "supervisor", "diagnostician": "supervisor",
    "self_healer": "supervisor", "strategy_promoter": "supervisor",
    "scheduler": "supervisor",
}


def _fetch_recent_events(
    limit: int = 100,
    since_id: int = 0,
    graph_name: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch recent agent events from PostgreSQL."""
    from quantstack.db import db_conn

    clauses = ["id > %s"]
    params: list[Any] = [since_id]

    if graph_name:
        clauses.append("graph_name = %s")
        params.append(graph_name)

    where = " AND ".join(clauses)
    params.append(limit)

    with db_conn() as conn:
        rows = conn.execute(
            f"""SELECT id, graph_name, node_name, agent_name, event_type,
                       content, metadata, cycle_number, created_at
                FROM agent_events
                WHERE {where}
                ORDER BY id ASC
                LIMIT %s""",
            params,
        ).fetchall()

    results = []
    for r in rows:
        graph = r[1] or "unknown"
        agent = r[3] or r[2] or ""
        # Fix team for old events tagged "unknown"
        if graph == "unknown" and agent in _AGENT_TEAM_FALLBACK:
            graph = _AGENT_TEAM_FALLBACK[agent]
        results.append({
            "id": r[0],
            "graph_name": graph,
            "node_name": r[2],
            "agent_name": agent,
            "event_type": r[4],
            "content": r[5],
            "metadata": r[6] if isinstance(r[6], dict) else {},
            "cycle_number": r[7],
            "created_at": r[8].isoformat() if r[8] else "",
        })
    return results


def _fetch_alerts(
    status: str = "open",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Fetch system alerts for dashboard display."""
    from quantstack.db import db_conn

    with db_conn() as conn:
        rows = conn.execute(
            """SELECT id, category, severity, status, source, title, detail,
                      metadata, created_at, acknowledged_at, resolved_at
               FROM system_alerts
               WHERE status = %s
               ORDER BY
                   CASE severity
                       WHEN 'emergency' THEN 1
                       WHEN 'critical' THEN 2
                       WHEN 'warning' THEN 3
                       WHEN 'info' THEN 4
                   END,
                   created_at DESC
               LIMIT %s""",
            [status, limit],
        ).fetchall()

    return [
        {
            "id": r[0],
            "category": r[1],
            "severity": r[2],
            "status": r[3],
            "source": r[4],
            "title": r[5],
            "detail": r[6],
            "metadata": r[7] if isinstance(r[7], dict) else {},
            "created_at": r[8].isoformat() if r[8] else "",
            "acknowledged_at": r[9].isoformat() if r[9] else None,
            "resolved_at": r[10].isoformat() if r[10] else None,
        }
        for r in rows
    ]


def _fetch_graph_status() -> list[dict[str, Any]]:
    """Fetch latest checkpoint per graph."""
    from quantstack.db import db_conn

    with db_conn() as conn:
        rows = conn.execute("""
            SELECT DISTINCT ON (graph_name)
                   graph_name, cycle_number, status, duration_seconds, created_at
            FROM graph_checkpoints
            ORDER BY graph_name, created_at DESC
        """).fetchall()

    return [
        {
            "graph_name": r[0],
            "cycle_number": r[1],
            "status": r[2],
            "duration_seconds": round(r[3], 1),
            "last_run": r[4].isoformat() if r[4] else "",
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/api/events")
def get_events(
    limit: int = Query(default=100, le=500),
    since_id: int = Query(default=0),
    graph: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch recent agent events (polling endpoint)."""
    return _fetch_recent_events(limit=limit, since_id=since_id, graph_name=graph)


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    """Graph runner status overview."""
    return {
        "graphs": _fetch_graph_status(),
        "as_of": datetime.now().isoformat(),
    }


@app.get("/api/alerts")
def get_alerts(
    status: str = "open",
    limit: int = Query(default=20, le=100),
) -> list[dict[str, Any]]:
    """Return recent system alerts for dashboard display."""
    return _fetch_alerts(status=status, limit=limit)


@app.get("/api/stream")
async def stream_events(since_id: int = Query(default=0)):
    """SSE endpoint — streams new agent events as they arrive."""

    async def event_generator() -> AsyncGenerator[str, None]:
        last_id = since_id
        while True:
            events = _fetch_recent_events(limit=50, since_id=last_id)
            for event in events:
                last_id = event["id"]
                yield f"data: {json.dumps(event, default=str)}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Single-page dashboard with per-team chat windows."""
    return DASHBOARD_HTML


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QuantStack Agent Dashboard</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --surface2: #1c2128; --border: #30363d;
    --text: #c9d1d9; --dim: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --yellow: #d29922; --red: #f85149; --purple: #bc8cff;
    --orange: #f0883e;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text);
    height: 100vh; display: flex; flex-direction: column;
  }

  /* Header */
  header {
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 10px 20px; display: flex; align-items: center; gap: 16px;
    flex-shrink: 0;
  }
  header h1 { font-size: 15px; font-weight: 700; color: var(--accent); letter-spacing: 0.5px; }
  header .status { font-size: 11px; color: var(--dim); display: flex; align-items: center; gap: 4px; }
  .dot { width: 7px; height: 7px; border-radius: 50%; }
  .dot.live { background: var(--green); animation: pulse 2s infinite; }
  .dot.off { background: var(--dim); }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
  .stats { margin-left: auto; display: flex; gap: 16px; }
  .stat { font-size: 11px; color: var(--dim); }
  .stat b { color: var(--text); font-weight: 600; }

  /* Grid */
  .grid {
    display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr;
    flex: 1; gap: 1px; background: var(--border); overflow: hidden;
  }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; grid-template-rows: repeat(4, 1fr); } }

  /* Team pane */
  .team { background: var(--surface); display: flex; flex-direction: column; overflow: hidden; }
  .team-header {
    padding: 8px 14px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 10px; flex-shrink: 0;
    background: var(--surface2);
  }
  .team-header .icon { font-size: 14px; }
  .team-header h2 { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; }
  .team-header .badge {
    font-size: 10px; font-weight: 600; padding: 1px 7px; border-radius: 10px;
    background: var(--border); color: var(--dim); margin-left: auto;
  }
  .team-header .cycle { font-size: 10px; color: var(--dim); }

  /* Chat messages */
  .chat { flex: 1; overflow-y: auto; padding: 6px 10px; display: flex; flex-direction: column; gap: 4px; }
  .msg {
    font-size: 12px; line-height: 1.6; padding: 6px 10px; border-radius: 6px;
    background: rgba(255,255,255,0.02); border-left: 3px solid var(--border);
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 11px; word-wrap: break-word; overflow-wrap: anywhere;
  }
  .msg .head {
    display: flex; align-items: center; gap: 6px; margin-bottom: 2px;
    font-size: 10px; color: var(--dim);
  }
  .msg .head .icon { font-size: 11px; }
  .msg .head .name { font-weight: 600; color: var(--text); }
  .msg .head .type { opacity: 0.7; }
  .msg .head .time { margin-left: auto; font-size: 9px; }
  .msg .body { color: var(--text); }
  .msg .body .tool-name { color: var(--yellow); font-weight: 600; }
  .msg .body .tool-args { color: var(--dim); }

  /* Event type styling */
  .msg.agent_start { border-left-color: var(--accent); background: rgba(88,166,255,0.04); }
  .msg.agent_response { border-left-color: var(--green); background: rgba(63,185,80,0.04); }
  .msg.tool_call { border-left-color: var(--yellow); }
  .msg.node_complete { border-left-color: var(--purple); background: rgba(188,140,255,0.04); }
  .msg.error { border-left-color: var(--red); background: rgba(248,81,73,0.04); }
  .msg.system_alert { border-left-color: var(--red); background: rgba(248,81,73,0.06); }

  .empty { color: var(--dim); font-size: 12px; text-align: center; padding: 60px 20px; }
  .empty .icon { font-size: 24px; margin-bottom: 8px; }

  /* Scrollbar */
  .chat::-webkit-scrollbar { width: 5px; }
  .chat::-webkit-scrollbar-track { background: transparent; }
  .chat::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .chat::-webkit-scrollbar-thumb:hover { background: var(--dim); }
</style>
</head>
<body>
<header>
  <h1>QUANTSTACK</h1>
  <div class="status"><span class="dot live" id="conn-dot"></span> <span id="conn-status">Connecting...</span></div>
  <div class="stats">
    <div class="stat">Events: <b id="event-count">0</b></div>
    <div class="stat">Agents: <b id="agent-count">0</b></div>
  </div>
</header>
<div class="grid">
  <div class="team" id="team-research">
    <div class="team-header">
      <span class="icon">🔬</span>
      <h2 style="color: var(--accent)">Research</h2>
      <span class="cycle" id="cycle-research"></span>
      <span class="badge" id="badge-research">0</span>
    </div>
    <div class="chat" id="chat-research"><div class="empty"><div class="icon">🔬</div>Waiting for research agents...</div></div>
  </div>
  <div class="team" id="team-trading">
    <div class="team-header">
      <span class="icon">📈</span>
      <h2 style="color: var(--green)">Trading</h2>
      <span class="cycle" id="cycle-trading"></span>
      <span class="badge" id="badge-trading">0</span>
    </div>
    <div class="chat" id="chat-trading"><div class="empty"><div class="icon">📈</div>Paused (market closed)</div></div>
  </div>
  <div class="team" id="team-supervisor">
    <div class="team-header">
      <span class="icon">🛡️</span>
      <h2 style="color: var(--purple)">Supervisor</h2>
      <span class="cycle" id="cycle-supervisor"></span>
      <span class="badge" id="badge-supervisor">0</span>
    </div>
    <div class="chat" id="chat-supervisor"><div class="empty"><div class="icon">🛡️</div>Waiting for supervisor events...</div></div>
  </div>
  <div class="team" id="team-other">
    <div class="team-header">
      <span class="icon">🤖</span>
      <h2 style="color: var(--orange)">FinRL / Other</h2>
      <span class="badge" id="badge-other">0</span>
    </div>
    <div class="chat" id="chat-other"><div class="empty"><div class="icon">🤖</div>Waiting for events...</div></div>
  </div>
</div>

<script>
// Agent → team fallback for old "unknown" events
const AGENT_TEAMS = {
  'quant_researcher': 'research', 'ml_scientist': 'research',
  'position_monitor': 'trading', 'entry_scanner': 'trading',
  'exit_manager': 'trading', 'risk_assessor': 'trading',
  'daily_planner': 'trading', 'fund_manager': 'trading',
  'options_analyst': 'trading', 'trade_debater': 'trading',
  'reflector': 'trading', 'execution_manager': 'trading',
  'health_monitor': 'supervisor', 'diagnostician': 'supervisor',
  'self_healer': 'supervisor', 'strategy_promoter': 'supervisor',
  'scheduler': 'supervisor',
};

const counts = { research: 0, trading: 0, supervisor: 0, other: 0 };
const seenAgents = new Set();
let totalEvents = 0;

function getTeam(graphName, agentName) {
  if (graphName && graphName !== 'unknown') {
    if (graphName.includes('research')) return 'research';
    if (graphName.includes('trading')) return 'trading';
    if (graphName.includes('supervisor')) return 'supervisor';
  }
  return AGENT_TEAMS[agentName] || 'other';
}

function formatTime(isoStr) {
  if (!isoStr) return '';
  const d = new Date(isoStr);
  return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

const TYPE_CONFIG = {
  agent_start:    { icon: '▶', label: 'thinking' },
  agent_response: { icon: '✓', label: 'decided' },
  tool_call:      { icon: '⚙', label: 'tool' },
  node_complete:  { icon: '◆', label: 'done' },
  error:          { icon: '✗', label: 'error' },
  system_alert:   { icon: '!', label: 'ALERT' },
};

function formatContent(event) {
  const content = event.content || '';
  if (event.event_type === 'tool_call') {
    // Parse "tool_name({args})" format
    const match = content.match(/^(\w+)\((.+)\)$/s);
    if (match) {
      const args = match[2].length > 120 ? match[2].substring(0, 120) + '...' : match[2];
      return '<span class="tool-name">' + escapeHtml(match[1]) + '</span> <span class="tool-args">' + escapeHtml(args) + '</span>';
    }
  }
  // Truncate long responses
  const maxLen = event.event_type === 'agent_start' ? 200 : 300;
  const truncated = content.length > maxLen ? content.substring(0, maxLen) + '...' : content;
  return escapeHtml(truncated);
}

function addMessage(event) {
  const team = getTeam(event.graph_name, event.agent_name);
  const chatEl = document.getElementById('chat-' + team);
  if (!chatEl) return;

  const empty = chatEl.querySelector('.empty');
  if (empty) empty.remove();

  const cfg = TYPE_CONFIG[event.event_type] || { icon: '•', label: event.event_type };
  const agentName = event.agent_name || event.node_name || '?';

  const msg = document.createElement('div');
  msg.className = 'msg ' + (event.event_type || '');
  msg.innerHTML =
    '<div class="head">' +
      '<span class="icon">' + cfg.icon + '</span>' +
      '<span class="name">' + escapeHtml(agentName) + '</span>' +
      '<span class="type">' + cfg.label + '</span>' +
      '<span class="time">' + formatTime(event.created_at) + '</span>' +
    '</div>' +
    '<div class="body">' + formatContent(event) + '</div>';

  chatEl.appendChild(msg);
  chatEl.scrollTop = chatEl.scrollHeight;

  // Keep max 300 messages per pane
  while (chatEl.children.length > 300) chatEl.removeChild(chatEl.firstChild);

  counts[team]++;
  totalEvents++;
  seenAgents.add(agentName);
  document.getElementById('badge-' + team).textContent = counts[team];
  document.getElementById('event-count').textContent = totalEvents;
  document.getElementById('agent-count').textContent = seenAgents.size;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Load graph status for cycle counts
function loadStatus() {
  fetch('/api/status').then(r => r.json()).then(data => {
    for (const g of (data.graphs || [])) {
      const el = document.getElementById('cycle-' + g.graph_name);
      if (el) el.textContent = 'cycle ' + g.cycle_number + ' · ' + g.duration_seconds + 's';
    }
  }).catch(() => {});
}
setInterval(loadStatus, 15000);
loadStatus();

// SSE connection with reconnect
function connect(startId) {
  const status = document.getElementById('conn-status');
  const dot = document.getElementById('conn-dot');
  const es = new EventSource('/api/stream?since_id=' + startId);
  let lastId = startId;

  es.onopen = () => {
    status.textContent = 'Live';
    dot.className = 'dot live';
  };
  es.onmessage = (e) => {
    const event = JSON.parse(e.data);
    lastId = event.id;
    addMessage(event);
  };
  es.onerror = () => {
    status.textContent = 'Reconnecting...';
    dot.className = 'dot off';
    es.close();
    setTimeout(() => connect(lastId), 3000);
  };
}

// Load history then stream
fetch('/api/events?limit=300')
  .then(r => r.json())
  .then(events => {
    events.forEach(addMessage);
    const lastId = events.length > 0 ? events[events.length - 1].id : 0;
    connect(lastId);
  })
  .catch(() => connect(0));
</script>
</body>
</html>
"""


def main():
    """Entry point: python -m quantstack.dashboard.app"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8421)


if __name__ == "__main__":
    main()
