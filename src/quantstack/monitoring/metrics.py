# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Prometheus metrics for the QuantStack trading system.

Instruments:
  - trades_executed_total        counter  symbol, side, speed (tick|minute)
  - risk_rejections_total        counter  violation_type
  - agent_latency_seconds        histogram  agent_name
  - signal_staleness_seconds     gauge    symbol
  - portfolio_nav_dollars        gauge
  - daily_pnl_dollars            gauge
  - kill_switch_active           gauge    (1 = active, 0 = inactive)
  - tick_executor_lag_seconds    histogram  (tick arrival → order submit)

All metrics are lazily registered on first use.

Usage:
    from quantstack.monitoring.metrics import record_fill, record_risk_rejection

    # After a fill in TickExecutor
    record_fill(symbol="SPY", side="buy", speed="tick")

    # After a risk rejection
    record_risk_rejection(violation_type="daily_loss_limit")

    # NAV / P&L (call from background flusher, e.g. every 30s)
    record_nav(nav_dollars=101_234.56)
    record_daily_pnl(pnl_dollars=+1_234.56)

    # Kill switch state (call when toggled)
    record_kill_switch_active(True)

    # Signal staleness (call from analysis plane each cycle)
    record_signal_staleness(symbol="SPY", staleness_seconds=45.2)

    # Agent latency (call after each LLM decision)
    record_agent_latency(agent_name="TrendIC", latency_seconds=2.34)

    # Tick executor hot-path lag
    record_tick_latency(latency_seconds=0.0012)

Exposition:
    Metrics are served at GET /metrics by the FastAPI server.
    The endpoint returns text/plain in Prometheus text exposition format.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

_trades_executed: Counter | None = None
_risk_rejections: Counter | None = None
_agent_latency: Histogram | None = None
_signal_staleness: Gauge | None = None
_portfolio_nav: Gauge | None = None
_daily_pnl: Gauge | None = None
_kill_switch: Gauge | None = None
_tick_lag: Histogram | None = None

# BLITZ mode metrics
_blitz_iterations: Counter | None = None
_blitz_duration: Histogram | None = None
_blitz_agents_spawned: Histogram | None = None
_blitz_agents_succeeded: Histogram | None = None
_blitz_symbols_complete: Gauge | None = None
_blitz_conflicts: Counter | None = None
_blitz_strategies: Counter | None = None


def _init_metrics() -> None:
    """Create all metrics once.  Safe to call multiple times."""
    global _trades_executed, _risk_rejections, _agent_latency
    global _signal_staleness, _portfolio_nav, _daily_pnl
    global _kill_switch, _tick_lag
    global _blitz_iterations, _blitz_duration, _blitz_agents_spawned
    global _blitz_agents_succeeded, _blitz_symbols_complete, _blitz_conflicts
    global _blitz_strategies
    if _trades_executed is not None:
        return

    _trades_executed = Counter(
        "quantstack_trades_executed_total",
        "Number of fills (rejected excluded) by symbol, side, and execution speed",
        ["symbol", "side", "speed"],
    )
    _risk_rejections = Counter(
        "quantstack_risk_rejections_total",
        "Number of orders rejected by the risk gate, by violation type",
        ["violation_type"],
    )
    _agent_latency = Histogram(
        "quantstack_agent_latency_seconds",
        "LLM agent decision latency in seconds",
        ["agent_name"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    )
    _signal_staleness = Gauge(
        "quantstack_signal_staleness_seconds",
        "Seconds since the most recent valid signal for this symbol",
        ["symbol"],
    )
    _portfolio_nav = Gauge(
        "quantstack_portfolio_nav_dollars",
        "Current total portfolio equity (cash + open positions at market)",
    )
    _daily_pnl = Gauge(
        "quantstack_daily_pnl_dollars",
        "Realized P&L for the current trading day",
    )
    _kill_switch = Gauge(
        "quantstack_kill_switch_active",
        "1 if the emergency kill switch is active, 0 otherwise",
    )
    _tick_lag = Histogram(
        "quantstack_tick_executor_lag_seconds",
        "Elapsed time from tick arrival to order submit in the hot path",
        buckets=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    )

    # BLITZ mode metrics
    _blitz_iterations = Counter(
        "quantstack_blitz_iterations_total",
        "Total BLITZ mode iterations completed"
    )
    _blitz_duration = Histogram(
        "quantstack_blitz_duration_seconds",
        "BLITZ iteration duration (seconds)",
        buckets=[30, 60, 120, 300, 600, 1200]  # 30s to 20min
    )
    _blitz_agents_spawned = Histogram(
        "quantstack_blitz_agents_spawned",
        "Number of agents spawned per BLITZ iteration",
        buckets=[3, 6, 9, 15, 21, 30]
    )
    _blitz_agents_succeeded = Histogram(
        "quantstack_blitz_agents_succeeded",
        "Number of agents that completed successfully",
        buckets=[0, 1, 3, 6, 9, 15, 21, 30]
    )
    _blitz_symbols_complete = Gauge(
        "quantstack_blitz_symbols_complete",
        "Symbols with complete 3-domain coverage (cumulative)"
    )
    _blitz_conflicts = Counter(
        "quantstack_blitz_conflicts_detected_total",
        "Cross-domain thesis conflicts detected",
        ["symbol"]
    )
    _blitz_strategies = Counter(
        "quantstack_blitz_strategies_registered_total",
        "Strategies registered by BLITZ agents",
        ["domain"]  # investment, swing, options
    )


# ---------------------------------------------------------------------------
# Public recording functions
# ---------------------------------------------------------------------------


def record_fill(symbol: str, side: str, speed: str = "tick") -> None:
    """Increment the fill counter.  speed in {"tick", "minute"}."""
    _init_metrics()
    _trades_executed.labels(symbol=symbol.upper(), side=side.lower(), speed=speed).inc()


def record_risk_rejection(violation_type: str) -> None:
    """Increment the risk-rejection counter."""
    _init_metrics()
    _risk_rejections.labels(violation_type=violation_type).inc()


def record_agent_latency(agent_name: str, latency_seconds: float) -> None:
    """Record a single LLM agent decision latency observation."""
    _init_metrics()
    _agent_latency.labels(agent_name=agent_name).observe(latency_seconds)


def record_signal_staleness(symbol: str, staleness_seconds: float) -> None:
    """Set the current staleness gauge for a symbol."""
    _init_metrics()
    _signal_staleness.labels(symbol=symbol.upper()).set(staleness_seconds)


def record_nav(nav_dollars: float) -> None:
    """Set the current portfolio NAV gauge."""
    _init_metrics()
    _portfolio_nav.set(nav_dollars)


def record_daily_pnl(pnl_dollars: float) -> None:
    """Set the current daily P&L gauge."""
    _init_metrics()
    _daily_pnl.set(pnl_dollars)


def record_kill_switch_active(active: bool) -> None:
    """Set the kill switch state gauge (1.0 = active, 0.0 = inactive)."""
    _init_metrics()
    _kill_switch.set(1.0 if active else 0.0)


def record_tick_latency(latency_seconds: float) -> None:
    """Record a single tick-executor hot-path latency observation."""
    _init_metrics()
    _tick_lag.observe(latency_seconds)


# ---------------------------------------------------------------------------
# BLITZ mode recording functions
# ---------------------------------------------------------------------------


def record_blitz_iteration(duration_seconds: float, agents_spawned: int, agents_succeeded: int) -> None:
    """Record a completed BLITZ iteration with metrics.

    Args:
        duration_seconds: Total time from start to completion
        agents_spawned: Number of agents launched
        agents_succeeded: Number of agents that completed successfully
    """
    _init_metrics()
    _blitz_iterations.inc()
    _blitz_duration.observe(duration_seconds)
    _blitz_agents_spawned.observe(agents_spawned)
    _blitz_agents_succeeded.observe(agents_succeeded)


def record_blitz_coverage(symbols_complete_count: int) -> None:
    """Update the cumulative count of symbols with complete 3-domain coverage.

    Args:
        symbols_complete_count: Number of symbols with investment + swing + options strategies
    """
    _init_metrics()
    _blitz_symbols_complete.set(symbols_complete_count)


def record_blitz_conflict(symbol: str) -> None:
    """Increment conflict counter when cross-domain thesis conflicts are detected.

    Args:
        symbol: Symbol with conflicting theses across domains
    """
    _init_metrics()
    _blitz_conflicts.labels(symbol=symbol.upper()).inc()


def record_blitz_strategy(domain: str) -> None:
    """Increment strategy registration counter by domain.

    Args:
        domain: Domain where strategy was registered (investment, swing, options)
    """
    _init_metrics()
    _blitz_strategies.labels(domain=domain.lower()).inc()


# ---------------------------------------------------------------------------
# Exposition helper (used by /metrics endpoint)
# ---------------------------------------------------------------------------


def get_metrics_text() -> str:
    """Return current metric values in Prometheus text exposition format."""
    _init_metrics()
    return generate_latest(REGISTRY).decode("utf-8")


def get_metrics_content_type() -> str:
    """Return the Content-Type header value for /metrics responses."""
    return CONTENT_TYPE_LATEST
