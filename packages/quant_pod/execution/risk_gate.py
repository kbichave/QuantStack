# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Hard risk controls — enforced in CODE, not in prompts.

This layer sits between every agent recommendation and the broker.
No agent, prompt, or instruction can bypass it.

Rules enforced:
  - Max position size per symbol (% of equity)
  - Max total notional exposure (% of equity)
  - Max single-name concentration
  - Daily loss limit (halt trading when breached)
  - Max leverage
  - Min liquidity (won't trade illiquid symbols)
  - Restricted symbol list (compliance-mandated)

Usage:
    gate = RiskGate()

    verdict = gate.check(
        symbol="SPY",
        side="buy",
        quantity=1000,
        current_price=450.0,
        daily_volume=80_000_000,
    )

    if verdict.approved:
        broker.execute(...)
    else:
        logger.warning(verdict.reason)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from threading import Lock

import yaml
from loguru import logger

from quant_pod.execution.portfolio_state import get_portfolio_state

# =============================================================================
# LIMITS CONFIG
# =============================================================================


@dataclass
class RiskLimits:
    """Configurable hard limits. Load from env or YAML."""

    # Per-symbol
    max_position_pct: float = 0.10  # 10% of equity per symbol
    max_position_notional: float = 20_000.0  # Hard $$ cap per position

    # Portfolio-level
    max_gross_exposure_pct: float = 1.50  # 150% gross (allows modest leverage)
    max_net_exposure_pct: float = 1.00  # 100% net long

    # Daily loss limit
    daily_loss_limit_pct: float = 0.02  # -2% of equity halts trading for the day

    # Liquidity
    min_daily_volume: int = 500_000  # Won't trade below 500k ADV
    max_participation_pct: float = 0.01  # Order <= 1% of ADV

    # Restricted list
    restricted_symbols: set[str] = field(default_factory=set)

    # ── Options-specific limits (v0.5.0) ─────────────────────────────────────
    # These only apply when instrument_type='options' is passed to check().
    # Equity checks are unchanged and run regardless.

    # Per-position: max premium at risk as % of equity
    # For debit structures: premium paid. For credit structures: max loss (spread_width - credit).
    max_premium_at_risk_pct: float = 0.02   # 2% of equity per options position

    # Portfolio total: max options premium outstanding as % of equity
    max_total_premium_pct: float = 0.08     # 8% of equity total options book

    # DTE bounds at entry
    min_dte_entry: int = 7    # No entries with < 7 DTE (gamma pins, binary outcomes)
    max_dte_entry: int = 60   # No far-dated speculative entries

    # ── Intraday limits (v0.6.0) ─────────────────────────────────────────────
    # TODO(strategy_breaker): enforcement lives in strategy_breaker.py, not check().
    # These fields are loaded from env and read by StrategyBreaker.should_halt().
    # Do not add enforcement here — check() is called per-trade; daily counters
    # belong in the StrategyBreaker which has session-level state.
    max_trades_per_day: int = 0  # 0 = unlimited; >0 = hard cap on daily orders
    entry_cutoff_minutes_before_close: int = 0  # 0 = disabled; >0 = no entries N min before close

    @classmethod
    def from_env(cls) -> RiskLimits:
        """Load limits from environment variables (override defaults)."""
        limits = cls()
        if v := os.getenv("RISK_MAX_POSITION_PCT"):
            limits.max_position_pct = float(v)
        if v := os.getenv("RISK_MAX_POSITION_NOTIONAL"):
            limits.max_position_notional = float(v)
        if v := os.getenv("RISK_DAILY_LOSS_LIMIT_PCT"):
            limits.daily_loss_limit_pct = float(v)
        if v := os.getenv("RISK_MIN_DAILY_VOLUME"):
            limits.min_daily_volume = int(v)
        if v := os.getenv("RISK_RESTRICTED_SYMBOLS"):
            limits.restricted_symbols = set(v.split(","))
        # Options limits
        if v := os.getenv("RISK_MAX_PREMIUM_AT_RISK_PCT"):
            limits.max_premium_at_risk_pct = float(v)
        if v := os.getenv("RISK_MAX_TOTAL_PREMIUM_PCT"):
            limits.max_total_premium_pct = float(v)
        if v := os.getenv("RISK_MIN_DTE_ENTRY"):
            limits.min_dte_entry = int(v)
        if v := os.getenv("RISK_MAX_DTE_ENTRY"):
            limits.max_dte_entry = int(v)
        # Intraday limits
        if v := os.getenv("RISK_MAX_TRADES_PER_DAY"):
            limits.max_trades_per_day = int(v)
        if v := os.getenv("RISK_ENTRY_CUTOFF_MINUTES"):
            limits.entry_cutoff_minutes_before_close = int(v)
        return limits

    @classmethod
    def from_yaml(cls, path: str) -> RiskLimits:
        """Load limits from a YAML config file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        limits = cls()
        for key, val in (data.get("risk_limits") or {}).items():
            if hasattr(limits, key):
                setattr(limits, key, val)
        return limits


# =============================================================================
# VERDICT
# =============================================================================


@dataclass
class RiskViolation:
    """Description of a single limit breach."""

    rule: str
    limit: float
    actual: float
    description: str


@dataclass
class RiskVerdict:
    """Result of a risk gate check."""

    approved: bool
    violations: list[RiskViolation] = field(default_factory=list)
    # Callers MUST use approved_quantity, not the original order quantity.
    # When approved=True and approved_quantity < requested, the order was scaled down.
    approved_quantity: int | None = None

    @property
    def reason(self) -> str:
        if self.approved:
            return "APPROVED"
        return " | ".join(v.description for v in self.violations)


# =============================================================================
# RISK GATE
# =============================================================================


class RiskGate:
    """
    Hard stop enforcement at execution boundary.

    Every trade MUST pass through check() before reaching the broker.
    Violations are logged and the trade is rejected or scaled.

    Daily halt state is persisted to a sentinel file so it survives process
    restarts — a crashed + restarted process will not silently resume trading
    on a day that had already been halted.
    """

    DAILY_HALT_SENTINEL = Path(
        os.getenv("DAILY_HALT_SENTINEL", "~/.quant_pod/DAILY_HALT_ACTIVE")
    ).expanduser()

    _lock = Lock()

    def __init__(
        self,
        limits: RiskLimits | None = None,
        portfolio: PortfolioState | None = None,  # type: ignore[name-defined]  # noqa: F821
    ):
        self.limits = limits or RiskLimits.from_env()
        # Accept injected PortfolioState (preferred) or fall back to singleton
        self._portfolio = portfolio if portfolio is not None else get_portfolio_state()
        self._daily_halted: date | None = None
        # Recover halt state from previous session if sentinel is present
        self._load_halt_sentinel()
        logger.info(
            f"RiskGate initialized | max_pos={self.limits.max_position_pct:.0%} "
            f"| daily_loss_limit={self.limits.daily_loss_limit_pct:.0%} "
            f"| halted={self.is_halted()}"
        )

    def _write_halt_sentinel(self) -> None:
        """Persist halt state so a restart doesn't silently resume trading."""
        self.DAILY_HALT_SENTINEL.parent.mkdir(parents=True, exist_ok=True)
        with open(self.DAILY_HALT_SENTINEL, "w") as f:
            f.write(f"halted_date={date.today().isoformat()}\n")
            f.write(f"written_at={datetime.now().isoformat()}\n")

    def _delete_halt_sentinel(self) -> None:
        """Remove the sentinel file atomically."""
        try:
            self.DAILY_HALT_SENTINEL.unlink()
        except FileNotFoundError:
            pass  # Already absent — that's fine

    def _load_halt_sentinel(self) -> None:
        """On init, check if a prior halt sentinel exists for today."""
        if not self.DAILY_HALT_SENTINEL.exists():
            return
        try:
            lines = self.DAILY_HALT_SENTINEL.read_text().strip().splitlines()
            data = {}
            for line in lines:
                if "=" in line:
                    k, v = line.split("=", 1)
                    data[k.strip()] = v.strip()
            halted_date = date.fromisoformat(data.get("halted_date", ""))
            if halted_date == date.today():
                self._daily_halted = halted_date
                logger.critical(
                    "[RISK] Daily halt sentinel found from previous session — "
                    "trading is HALTED for today. Call reset_daily_halt() to resume."
                )
        except Exception as e:
            # Treat an unreadable sentinel as active to be safe
            self._daily_halted = date.today()
            logger.warning(
                f"[RISK] Could not parse daily halt sentinel: {e} — treating as active halt"
            )

    # -------------------------------------------------------------------------
    # Main check
    # -------------------------------------------------------------------------

    def check(
        self,
        symbol: str,
        side: str,
        quantity: int,
        current_price: float,
        daily_volume: int = 0,
        instrument_type: str = "equity",
        premium_at_risk: float = 0.0,
        dte: int = 0,
    ) -> RiskVerdict:
        """
        Run all risk checks. Returns RiskVerdict (approved / rejected / scaled).

        For equity orders (instrument_type='equity', default):
            standard notional-based checks apply.

        For options orders (instrument_type='options'):
            - DTE bounds checked (min_dte_entry ≤ dte ≤ max_dte_entry)
            - premium_at_risk checked vs per-position limit (max_premium_at_risk_pct)
            - Equity notional checks (position size, gross exposure) are SKIPPED —
              options notional is not meaningful for risk sizing
            - Daily halt, restricted symbols, and liquidity checks still apply

        Args:
            symbol: Ticker symbol (underlying for options)
            side: "buy" or "sell"
            quantity: Number of shares (equity) or contracts (options)
            current_price: Latest underlying price
            daily_volume: Average daily volume of the underlying
            instrument_type: "equity" (default) or "options"
            premium_at_risk: For options only — total $ at risk for this position.
                For debit structures: premium paid. For credit structures: max loss
                (spread_width × contracts × 100 - credit received).
            dte: For options only — days to expiration at entry.
        """
        violations: list[RiskViolation] = []
        snapshot = self._portfolio.get_snapshot()

        # -- 1. Daily halt check (fast path — checked before all other work)
        if self._daily_halted == date.today():
            violations.append(
                RiskViolation(
                    rule="daily_loss_halt",
                    limit=0,
                    actual=0,
                    description="Trading halted for today — daily loss limit was breached",
                )
            )
            return RiskVerdict(approved=False, violations=violations)

        # -- 2. Restricted symbol
        if symbol.upper() in self.limits.restricted_symbols:
            violations.append(
                RiskViolation(
                    rule="restricted_symbol",
                    limit=0,
                    actual=0,
                    description=f"{symbol} is on the restricted list",
                )
            )
            return RiskVerdict(approved=False, violations=violations)

        # -- 3. Unknown volume: reject rather than skip liquidity checks.
        #    Silently skipping when volume=0 allows trading illiquid names through.
        if daily_volume == 0:
            violations.append(
                RiskViolation(
                    rule="unknown_volume",
                    limit=self.limits.min_daily_volume,
                    actual=0,
                    description=(
                        f"Cannot trade {symbol} — daily volume not provided. "
                        "Supply ADV from market data before submitting orders."
                    ),
                )
            )
            return RiskVerdict(approved=False, violations=violations)

        # -- 4. Daily loss limit
        if snapshot.total_equity > 0:
            daily_loss_pct = abs(min(0, snapshot.daily_pnl)) / snapshot.total_equity
            if daily_loss_pct >= self.limits.daily_loss_limit_pct:
                with self._lock:
                    self._daily_halted = date.today()
                    self._write_halt_sentinel()
                violations.append(
                    RiskViolation(
                        rule="daily_loss_limit",
                        limit=self.limits.daily_loss_limit_pct,
                        actual=daily_loss_pct,
                        description=(
                            f"Daily loss {daily_loss_pct:.1%} >= "
                            f"limit {self.limits.daily_loss_limit_pct:.1%} — HALT"
                        ),
                    )
                )
                return RiskVerdict(approved=False, violations=violations)

        # -- 5. Liquidity check
        if daily_volume < self.limits.min_daily_volume:
            violations.append(
                RiskViolation(
                    rule="min_daily_volume",
                    limit=self.limits.min_daily_volume,
                    actual=daily_volume,
                    description=(
                        f"{symbol} ADV {daily_volume:,} < minimum {self.limits.min_daily_volume:,}"
                    ),
                )
            )

        # -- 6. Participation rate: cap the order, don't reject.
        #    Market-impact scaling is a quantity adjustment, not a hard veto.
        #    Callers must read approved_quantity — never use the original quantity.
        participation = quantity / daily_volume
        if participation > self.limits.max_participation_pct:
            max_qty = int(daily_volume * self.limits.max_participation_pct)
            logger.warning(
                f"[RISK] {symbol} participation {participation:.1%} > "
                f"{self.limits.max_participation_pct:.1%}; "
                f"capping order {quantity:,} → {max_qty:,}"
            )
            quantity = max_qty

        if not violations and instrument_type == "options":
            # -- Options path: DTE and premium-at-risk checks.
            #    Equity notional checks (steps 7-8) are skipped for options.
            equity = snapshot.total_equity or 100_000.0

            # -- 7a. DTE bounds
            if dte > 0:
                if dte < self.limits.min_dte_entry:
                    violations.append(
                        RiskViolation(
                            rule="options_min_dte",
                            limit=self.limits.min_dte_entry,
                            actual=dte,
                            description=(
                                f"{symbol} option DTE {dte} < minimum {self.limits.min_dte_entry} "
                                "— gamma risk too high near expiry"
                            ),
                        )
                    )
                elif dte > self.limits.max_dte_entry:
                    violations.append(
                        RiskViolation(
                            rule="options_max_dte",
                            limit=self.limits.max_dte_entry,
                            actual=dte,
                            description=(
                                f"{symbol} option DTE {dte} > maximum {self.limits.max_dte_entry} "
                                "— no far-dated speculative entries"
                            ),
                        )
                    )

            # -- 7b. Per-position premium at risk
            if premium_at_risk > 0:
                max_premium = equity * self.limits.max_premium_at_risk_pct
                if premium_at_risk > max_premium:
                    violations.append(
                        RiskViolation(
                            rule="options_premium_at_risk",
                            limit=max_premium,
                            actual=premium_at_risk,
                            description=(
                                f"{symbol} options premium at risk ${premium_at_risk:,.0f} > "
                                f"limit ${max_premium:,.0f} "
                                f"({self.limits.max_premium_at_risk_pct:.0%} of equity)"
                            ),
                        )
                    )

            if violations:
                for v in violations:
                    logger.warning(f"[RISK] OPTIONS VIOLATION [{v.rule}]: {v.description}")
                return RiskVerdict(approved=False, violations=violations)

            return RiskVerdict(approved=True, approved_quantity=quantity)

        if not violations:
            # -- 7. Per-symbol position size (equity path)
            equity = snapshot.total_equity or 100_000.0
            order_notional = quantity * current_price

            existing_pos = self._portfolio.get_position(symbol)
            existing_notional = abs(existing_pos.quantity) * current_price if existing_pos else 0.0
            new_total_notional = existing_notional + order_notional

            max_notional_by_pct = equity * self.limits.max_position_pct
            max_notional = min(max_notional_by_pct, self.limits.max_position_notional)

            if new_total_notional > max_notional:
                allowed_additional = max(0.0, max_notional - existing_notional)
                scaled_qty = int(allowed_additional / current_price) if current_price > 0 else 0

                if scaled_qty <= 0:
                    violations.append(
                        RiskViolation(
                            rule="max_position_size",
                            limit=max_notional,
                            actual=new_total_notional,
                            description=(
                                f"{symbol} position would reach "
                                f"${new_total_notional:,.0f} vs "
                                f"limit ${max_notional:,.0f} — VETO"
                            ),
                        )
                    )
                else:
                    logger.warning(
                        f"[RISK] {symbol} position scaled: {quantity} → {scaled_qty} "
                        f"(${new_total_notional:,.0f} → ${scaled_qty * current_price:,.0f})"
                    )
                    return RiskVerdict(
                        approved=True,
                        violations=[],
                        approved_quantity=scaled_qty,
                    )

            # -- 8. Gross exposure
            positions = self._portfolio.get_positions()
            current_gross = sum(abs(p.quantity) * p.current_price for p in positions)
            new_gross = current_gross + order_notional
            max_gross = equity * self.limits.max_gross_exposure_pct

            if new_gross > max_gross:
                violations.append(
                    RiskViolation(
                        rule="max_gross_exposure",
                        limit=max_gross,
                        actual=new_gross,
                        description=(
                            f"Gross exposure would reach ${new_gross:,.0f} "
                            f"({new_gross / equity:.0%}) vs limit "
                            f"${max_gross:,.0f} ({self.limits.max_gross_exposure_pct:.0%})"
                        ),
                    )
                )

        if violations:
            for v in violations:
                logger.warning(f"[RISK] VIOLATION [{v.rule}]: {v.description}")
            return RiskVerdict(approved=False, violations=violations)

        return RiskVerdict(approved=True, approved_quantity=quantity)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def is_halted(self) -> bool:
        """True if trading is halted today due to daily loss limit."""
        if self._daily_halted == date.today():
            return True
        # Cross-process check: another process may have written the sentinel
        if self.DAILY_HALT_SENTINEL.exists():
            self._load_halt_sentinel()
            return self._daily_halted == date.today()
        return False

    def reset_daily_halt(self) -> None:
        """Manually reset the daily halt (ops use only)."""
        with self._lock:
            self._daily_halted = None
            self._delete_halt_sentinel()
        logger.info("[RISK] Daily halt manually reset")

    def add_restricted(self, symbol: str) -> None:
        """Add a symbol to the restricted list at runtime."""
        self.limits.restricted_symbols.add(symbol.upper())
        logger.info(f"[RISK] {symbol} added to restricted list")

    def remove_restricted(self, symbol: str) -> None:
        """Remove a symbol from the restricted list."""
        self.limits.restricted_symbols.discard(symbol.upper())
        logger.info(f"[RISK] {symbol} removed from restricted list")


# Singleton
_risk_gate: RiskGate | None = None


def get_risk_gate(limits: RiskLimits | None = None) -> RiskGate:
    """Get the singleton RiskGate instance."""
    global _risk_gate
    if _risk_gate is None:
        _risk_gate = RiskGate(limits=limits)
    return _risk_gate
