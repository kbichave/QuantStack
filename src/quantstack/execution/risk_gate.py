# Copyright 2024 QuantStack Contributors
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
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

import yaml
from loguru import logger

from quantstack.config.timeframes import Timeframe
from quantstack.data.storage import DataStore
from quantstack.db import PgConnection, pg_conn
from quantstack.core.risk.options_risk import PortfolioGreeksManager
from quantstack.execution.liquidity_model import LiquidityModel, LiquidityVerdict
from quantstack.execution.portfolio_state import get_portfolio_state
from quantstack.holding_period import HoldingType
from quantstack.universe import INITIAL_LIQUID_UNIVERSE, Sector

# Sector-to-ETF mapping for correlation proxy fallback
SECTOR_ETF_MAP: dict[Sector, str] = {
    Sector.TECHNOLOGY: "XLK",
    Sector.HEALTHCARE: "XLV",
    Sector.FINANCIALS: "XLF",
    Sector.ENERGY: "XLE",
    Sector.CONSUMER_DISCRETIONARY: "XLY",
    Sector.CONSUMER_STAPLES: "XLP",
    Sector.INDUSTRIALS: "XLI",
    Sector.MATERIALS: "XLB",
    Sector.REAL_ESTATE: "XLRE",
    Sector.UTILITIES: "XLU",
    Sector.COMMUNICATION: "XLC",
}

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
    max_premium_at_risk_pct: float = 0.02  # 2% of equity per options position

    # Portfolio total: max options premium outstanding as % of equity
    max_total_premium_pct: float = 0.08  # 8% of equity total options book

    # DTE bounds at entry
    min_dte_entry: int = 7  # No entries with < 7 DTE (gamma pins, binary outcomes)
    max_dte_entry: int = 60  # No far-dated speculative entries

    # ── Pre-trade portfolio checks (phase 4) ───────────────────────────────
    max_pretrade_correlation: float = 0.70
    max_daily_heat_pct: float = 0.30
    max_sector_concentration_pct: float = 0.40

    # ── Intraday limits (v0.6.0) ─────────────────────────────────────────────
    # TODO(strategy_breaker): enforcement lives in strategy_breaker.py, not check().
    # These fields are loaded from env and read by StrategyBreaker.should_halt().
    # Do not add enforcement here — check() is called per-trade; daily counters
    # belong in the StrategyBreaker which has session-level state.
    max_trades_per_day: int = 0  # 0 = unlimited; >0 = hard cap on daily orders
    entry_cutoff_minutes_before_close: int = (
        0  # 0 = disabled; >0 = no entries N min before close
    )

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
        # Pre-trade portfolio checks
        if v := os.getenv("RISK_MAX_PRETRADE_CORRELATION"):
            limits.max_pretrade_correlation = float(v)
        if v := os.getenv("RISK_MAX_DAILY_HEAT_PCT"):
            limits.max_daily_heat_pct = float(v)
        if v := os.getenv("RISK_MAX_SECTOR_CONCENTRATION_PCT"):
            limits.max_sector_concentration_pct = float(v)
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
# EXECUTION QUALITY LOOKUP
# =============================================================================


def get_execution_quality_scalar(symbol: str, conn: PgConnection) -> float:
    """Look up the most recent execution quality scalar for *symbol*.

    Queries the ``symbol_execution_quality`` table (populated by the nightly
    TCA feedback job) and returns the ``quality_scalar`` column.  If no row
    exists for the symbol, returns **1.0** (no adjustment).

    Args:
        symbol: Ticker symbol to look up.
        conn: An open :class:`PgConnection` (caller manages lifecycle).

    Returns:
        Quality scalar (e.g. 1.1, 1.0, 0.7, 0.5).
    """
    row = conn.execute(
        "SELECT quality_scalar FROM symbol_execution_quality "
        "WHERE symbol = %s ORDER BY week_ending DESC LIMIT 1",
        [symbol.upper()],
    ).fetchone()
    if row is not None:
        return float(row[0])
    return 1.0


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
        os.getenv("DAILY_HALT_SENTINEL", "~/.quantstack/DAILY_HALT_ACTIVE")
    ).expanduser()

    _lock = Lock()

    def __init__(
        self,
        limits: RiskLimits | None = None,
        portfolio: PortfolioState | None = None,  # type: ignore[name-defined]  # noqa: F821
        greeks_manager: PortfolioGreeksManager | None = None,
    ):
        self.limits = limits or RiskLimits.from_env()
        # Accept injected PortfolioState (preferred) or fall back to singleton
        self._portfolio = portfolio if portfolio is not None else get_portfolio_state()
        self._greeks_manager = greeks_manager or PortfolioGreeksManager()
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
        strategy_status: str = "live",
        holding_type: HoldingType | None = None,
        proposed_delta: float = 0.0,
        proposed_gamma: float = 0.0,
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

        # -- 0. Trading window check: reject exposure-increasing orders outside market hours
        window_verdict = self._check_market_hours(symbol, side)
        if window_verdict is not None:
            return window_verdict

        snapshot = self._portfolio.get_snapshot()
        if snapshot is None:
            return RiskVerdict(
                approved=False,
                violations=[
                    RiskViolation(
                        rule="portfolio_unavailable",
                        limit=0,
                        actual=0,
                        description="Portfolio snapshot unavailable — cannot evaluate risk",
                    )
                ],
            )

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

        # -- 1b. Holding period check (fast rejection before expensive DB lookups)
        if holding_type is not None:
            from quantstack.holding_period import HoldingPeriodManager

            if not HoldingPeriodManager().is_allowed(holding_type, "trading"):
                violations.append(
                    RiskViolation(
                        rule="holding_period",
                        limit=0,
                        actual=0,
                        description=(
                            f"holding_period_not_allowed: {holding_type.value} "
                            f"not in TRADING_WINDOW"
                        ),
                    )
                )
                return RiskVerdict(approved=False, violations=violations)

        # -- 1c. Trading window check — instrument type + DTE/hold-days gating
        try:
            from quantstack.config.settings import get_trading_window_settings
            from quantstack.trading_window import is_trade_allowed

            tw_settings = get_trading_window_settings()
            trading_windows = tw_settings.trading_windows

            if not is_trade_allowed(
                instrument_type,
                trading_windows,
                dte=dte if instrument_type == "options" else None,
            ):
                violations.append(
                    RiskViolation(
                        rule="trading_window",
                        limit=0,
                        actual=0,
                        description=(
                            f"trading_window_rejected: {instrument_type}"
                            f"{f' DTE={dte}' if instrument_type == 'options' else ''}"
                            f" not permitted by TRADING_WINDOW"
                        ),
                    )
                )
                return RiskVerdict(approved=False, violations=violations)
        except Exception as e:
            logger.warning(f"[RISK] Trading window check failed: {e} — allowing trade")

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

        # -- 3. Unknown/invalid volume: reject rather than skip liquidity checks.
        #    Silently skipping when volume=0 allows trading illiquid names through.
        if daily_volume <= 0:
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

        # -- 5b. Pre-trade liquidity model check (depth-based gating).
        #    Runs after volume validation so we know daily_volume is positive.
        #    REJECT = hard veto; SCALE_DOWN = reduce quantity before participation cap.
        if not violations:
            try:
                liq_model = LiquidityModel(daily_volumes={symbol: daily_volume})
                liq_result = liq_model.pre_trade_check(symbol, quantity)
                if liq_result.verdict == LiquidityVerdict.REJECT:
                    violations.append(
                        RiskViolation(
                            rule="liquidity_depth",
                            limit=0,
                            actual=0,
                            description=liq_result.reason,
                        )
                    )
                    return RiskVerdict(approved=False, violations=violations)
                if liq_result.verdict == LiquidityVerdict.SCALE_DOWN:
                    if liq_result.recommended_quantity is not None:
                        logger.warning(
                            f"[RISK] {symbol} liquidity scale-down: "
                            f"{quantity} → {liq_result.recommended_quantity} shares"
                        )
                        quantity = liq_result.recommended_quantity
            except Exception as exc:
                logger.debug(f"[RISK] liquidity model check failed: {exc}")

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

        # -- 6b. Execution quality scalar: penalise symbols with poor fills.
        #    Applied AFTER ADV cap so both adjustments compose multiplicatively.
        try:
            with pg_conn() as conn:
                quality_scalar = get_execution_quality_scalar(symbol, conn)
        except Exception as exc:
            logger.debug(f"[RISK] execution quality lookup failed: {exc}")
            quality_scalar = 1.0

        if quality_scalar != 1.0:
            pre_quality_qty = quantity
            quantity = max(1, round(quantity * quality_scalar))
            logger.info(
                f"[RISK] {symbol} execution quality scalar {quality_scalar:.2f}: "
                f"{pre_quality_qty} → {quantity} shares"
            )

        # -- 6c. Macro stress scalar: reduce size under stressed macro conditions.
        #    Applied AFTER both ADV cap and execution quality scalar.
        try:
            from quantstack.core.signals.alt_data_normalizer import (
                compute_macro_stress,
                get_macro_stress_scalar,
            )
            with pg_conn() as conn:
                stress_score = compute_macro_stress(conn)
            macro_scalar = get_macro_stress_scalar(stress_score)
        except Exception as exc:
            logger.debug(f"[RISK] macro stress lookup failed: {exc}")
            macro_scalar = 1.0
            stress_score = 0.0

        if macro_scalar != 1.0:
            pre_macro_qty = quantity
            quantity = max(1, round(quantity * macro_scalar))
            logger.info(
                f"[RISK] Macro stress scalar {macro_scalar:.2f} "
                f"(score {stress_score:.2f}): {pre_macro_qty} → {quantity} shares"
            )

        # -- 6d. Pre-trade portfolio-level checks (equity path only, new entries only)
        if not violations and instrument_type == "equity":
            # Determine is_reducing early for the guard
            existing_pos = self._portfolio.get_position(symbol)
            _is_reducing = False
            if existing_pos:
                existing_long = existing_pos.quantity > 0
                order_is_sell = side.upper() in ("SELL", "SHORT")
                _is_reducing = (existing_long and order_is_sell) or (
                    not existing_long and not order_is_sell
                )

            if not _is_reducing:
                _order_notional = quantity * current_price
                _equity = snapshot.total_equity or 100_000.0

                corr_violations = self._check_pretrade_correlation(symbol, current_price)
                violations.extend(corr_violations)

                heat_violations = self._check_heat_budget(_order_notional, _equity)
                violations.extend(heat_violations)

                sector_violations = self._check_sector_concentration(
                    symbol, _order_notional, _equity,
                )
                violations.extend(sector_violations)

                if violations:
                    for v in violations:
                        logger.warning(
                            f"[RISK] PRETRADE VIOLATION [{v.rule}]: {v.description}"
                        )
                    return RiskVerdict(approved=False, violations=violations)

        if not violations and instrument_type == "options":
            # -- Options path: DTE and premium-at-risk checks.
            #    Equity notional checks (steps 7-8) are skipped for options.
            equity = snapshot.total_equity or 100_000.0

            # -- 7a. DTE bounds — window-derived bounds widen the static defaults,
            #    but static RiskLimits still act as an absolute backstop.
            effective_min_dte = self.limits.min_dte_entry
            effective_max_dte = self.limits.max_dte_entry
            try:
                from quantstack.config.settings import get_trading_window_settings
                from quantstack.trading_window import get_dte_bounds

                dte_bounds = get_dte_bounds(get_trading_window_settings().trading_windows)
                if dte_bounds is not None:
                    effective_min_dte = min(effective_min_dte, dte_bounds[0])
                    effective_max_dte = max(effective_max_dte, dte_bounds[1])
            except Exception:
                pass  # fall back to static limits

            if dte >= 0:
                if dte < effective_min_dte:
                    violations.append(
                        RiskViolation(
                            rule="options_min_dte",
                            limit=effective_min_dte,
                            actual=dte,
                            description=(
                                f"{symbol} option DTE {dte} < minimum {effective_min_dte} "
                                "— gamma risk too high near expiry"
                            ),
                        )
                    )
                elif dte > effective_max_dte:
                    violations.append(
                        RiskViolation(
                            rule="options_max_dte",
                            limit=effective_max_dte,
                            actual=dte,
                            description=(
                                f"{symbol} option DTE {dte} > maximum {effective_max_dte} "
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

            # -- 7c. Portfolio Greeks limits (delta/gamma/vega per-trade check)
            allowed, reason = self._greeks_manager.check_new_trade(
                proposed_delta=proposed_delta,
                proposed_gamma=proposed_gamma,
                symbol=symbol,
            )
            if not allowed:
                violations.append(
                    RiskViolation(
                        rule="options_greeks_limit",
                        limit=0,
                        actual=0,
                        description=f"{symbol} Greeks limit breach: {reason}",
                    )
                )

            if violations:
                for v in violations:
                    logger.warning(
                        f"[RISK] OPTIONS VIOLATION [{v.rule}]: {v.description}"
                    )
                return RiskVerdict(approved=False, violations=violations)

            return RiskVerdict(approved=True, approved_quantity=quantity)

        if not violations:
            # -- 7. Per-symbol position size (equity path)
            equity = snapshot.total_equity or 100_000.0

            # Scale down position size for unproven (forward_testing) strategies.
            # All hard limits are still enforced on the scaled quantity.
            if strategy_status == "forward_testing":
                scalar = float(os.getenv("FORWARD_TESTING_SIZE_SCALAR", "0.5"))
                original_qty = quantity
                quantity = max(1, int(quantity * scalar))
                if quantity != original_qty:
                    logger.info(
                        f"[RISK] {symbol} forward_testing scalar {scalar:.0%}: "
                        f"{original_qty} → {quantity} shares"
                    )

            order_notional = quantity * current_price

            existing_pos = self._portfolio.get_position(symbol)
            existing_notional = (
                abs(existing_pos.quantity) * current_price if existing_pos else 0.0
            )

            # Determine if this order reduces or increases position exposure.
            # A sell against a long (or buy against a short) is a position
            # reduction and should not be penalised by the position limit.
            is_reducing = False
            if existing_pos:
                existing_long = existing_pos.quantity > 0
                order_is_sell = side.upper() in ("SELL", "SHORT")
                is_reducing = (existing_long and order_is_sell) or (
                    not existing_long and not order_is_sell
                )

            if is_reducing:
                new_total_notional = max(0.0, existing_notional - order_notional)
            else:
                new_total_notional = existing_notional + order_notional

            max_notional_by_pct = equity * self.limits.max_position_pct
            max_notional = min(max_notional_by_pct, self.limits.max_position_notional)

            if new_total_notional > max_notional:
                allowed_additional = max(0.0, max_notional - existing_notional)
                scaled_qty = (
                    int(allowed_additional / current_price) if current_price > 0 else 0
                )

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
            new_gross = current_gross + (-order_notional if is_reducing else order_notional)
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
    # Pre-trade portfolio checks (phase 4, section 12)
    # -------------------------------------------------------------------------

    def _check_pretrade_correlation(
        self, symbol: str, current_price: float,
    ) -> list[RiskViolation]:
        """Check correlation of candidate symbol with existing positions.

        Falls back to sector ETF proxy when candidate has <20 days history.
        Fails closed on data unavailability.
        """
        try:
            positions = self._portfolio.get_positions()
            if not positions:
                return []

            store = DataStore()
            candidate_df = store.load_ohlcv(symbol, Timeframe.D1)

            # Check if we need sector proxy fallback
            use_proxy = candidate_df is None or len(candidate_df) < 20
            proxy_symbol = None
            if use_proxy:
                uni_sym = INITIAL_LIQUID_UNIVERSE.get(symbol)
                if uni_sym and uni_sym.sector in SECTOR_ETF_MAP:
                    proxy_symbol = SECTOR_ETF_MAP[uni_sym.sector]
                    candidate_df = store.load_ohlcv(proxy_symbol, Timeframe.D1)

            if candidate_df is None or len(candidate_df) < 20:
                return [RiskViolation(
                    rule="pretrade_correlation_data_missing",
                    limit=self.limits.max_pretrade_correlation,
                    actual=0,
                    description=(
                        f"Insufficient price history for {symbol} correlation check "
                        f"— fail closed"
                    ),
                )]

            candidate_returns = candidate_df["close"].pct_change().dropna().tail(30)

            for pos in positions:
                pos_df = store.load_ohlcv(pos.symbol, Timeframe.D1)
                if pos_df is None or len(pos_df) < 20:
                    continue
                pos_returns = pos_df["close"].pct_change().dropna().tail(30)
                common_idx = candidate_returns.index.intersection(pos_returns.index)
                if len(common_idx) < 20:
                    continue
                corr = candidate_returns.loc[common_idx].corr(pos_returns.loc[common_idx])
                if corr >= self.limits.max_pretrade_correlation:
                    return [RiskViolation(
                        rule="pretrade_correlation",
                        limit=self.limits.max_pretrade_correlation,
                        actual=round(corr, 3),
                        description=(
                            f"{symbol} has {corr:.2f} correlation with existing "
                            f"position {pos.symbol} (limit: "
                            f"{self.limits.max_pretrade_correlation})"
                        ),
                    )]
            return []
        except Exception as exc:
            logger.warning(f"[RISK] pretrade correlation check failed: {exc}")
            return [RiskViolation(
                rule="pretrade_correlation_data_missing",
                limit=self.limits.max_pretrade_correlation,
                actual=0,
                description=f"Correlation check failed — fail closed: {exc}",
            )]

    def _check_heat_budget(
        self, order_notional: float, equity: float,
    ) -> list[RiskViolation]:
        """Check if daily notional deployment would exceed heat budget.

        System-wide query across all graph services sharing the same DB.
        """
        try:
            with pg_conn() as conn:
                row = conn.execute(
                    "SELECT COALESCE(SUM(quantity * avg_cost), 0) "
                    "FROM positions WHERE opened_at::date = CURRENT_DATE",
                ).fetchone()
            today_notional = float(row[0]) if row else 0.0
        except Exception as exc:
            logger.warning(f"[RISK] heat budget query failed: {exc}")
            return [RiskViolation(
                rule="daily_heat_budget",
                limit=self.limits.max_daily_heat_pct,
                actual=0,
                description=f"Heat budget check failed — fail closed: {exc}",
            )]

        total = today_notional + order_notional
        heat_pct = total / equity if equity > 0 else 1.0
        if heat_pct >= self.limits.max_daily_heat_pct:
            return [RiskViolation(
                rule="daily_heat_budget",
                limit=self.limits.max_daily_heat_pct,
                actual=round(heat_pct, 4),
                description=(
                    f"Daily heat {heat_pct:.1%} >= limit "
                    f"{self.limits.max_daily_heat_pct:.0%} "
                    f"(today ${today_notional:,.0f} + order ${order_notional:,.0f})"
                ),
            )]
        return []

    def _check_sector_concentration(
        self, symbol: str, order_notional: float, equity: float,
    ) -> list[RiskViolation]:
        """Check if adding position would exceed sector concentration limit."""
        try:
            uni_sym = INITIAL_LIQUID_UNIVERSE.get(symbol)
            if uni_sym is None:
                return []  # Unknown sector = unique sector, no concentration

            candidate_sector = uni_sym.sector
            if candidate_sector == Sector.ETF:
                return []  # ETFs don't contribute to sector concentration

            # Compute notional per sector across existing positions
            sector_notional: float = order_notional
            positions = self._portfolio.get_positions()
            for pos in positions:
                pos_uni = INITIAL_LIQUID_UNIVERSE.get(pos.symbol)
                if pos_uni and pos_uni.sector == candidate_sector:
                    sector_notional += abs(pos.quantity) * pos.current_price

            concentration = sector_notional / equity if equity > 0 else 1.0
            if concentration >= self.limits.max_sector_concentration_pct:
                return [RiskViolation(
                    rule="sector_concentration",
                    limit=self.limits.max_sector_concentration_pct,
                    actual=round(concentration, 4),
                    description=(
                        f"{candidate_sector.value} sector at {concentration:.1%} "
                        f">= limit {self.limits.max_sector_concentration_pct:.0%} "
                        f"after adding {symbol}"
                    ),
                )]
            return []
        except Exception as exc:
            logger.warning(f"[RISK] sector concentration check failed: {exc}")
            return [RiskViolation(
                rule="sector_concentration",
                limit=self.limits.max_sector_concentration_pct,
                actual=0,
                description=f"Sector concentration check failed — fail closed: {exc}",
            )]

    # -------------------------------------------------------------------------
    # Market hours check
    # -------------------------------------------------------------------------

    def _check_market_hours(self, symbol: str, side: str) -> RiskVerdict | None:
        """Reject exposure-increasing orders outside market hours.

        Returns None if the order is allowed (caller continues to other checks),
        or a RiskVerdict rejection if the order would increase exposure.

        Logic:
        - MARKET mode: all orders allowed (return None).
        - EXTENDED/OVERNIGHT/WEEKEND: look up current position for the symbol.
          If abs(new_qty) > abs(current_qty): reject.
          If abs(new_qty) <= abs(current_qty): allow (it's a closing trade).
        """
        from quantstack.runners import OperatingMode, get_operating_mode

        mode = get_operating_mode()
        if mode == OperatingMode.MARKET:
            return None

        # Determine whether this order increases or decreases exposure
        existing_pos = self._portfolio.get_position(symbol)
        current_qty = existing_pos.quantity if existing_pos else 0

        side_upper = side.upper()
        # Determine sign of the order
        if side_upper in ("BUY", "LONG"):
            # Buy increases qty
            would_increase = current_qty >= 0  # new long or adding to long
        elif side_upper in ("SELL", "SHORT"):
            # Sell decreases qty
            would_increase = current_qty <= 0  # new short or adding to short
        else:
            would_increase = True  # unknown side, fail closed

        if not would_increase:
            # This is a closing/reducing trade — allow it
            return None

        reason_map = {
            OperatingMode.EXTENDED: "Extended hours — no new entries",
            OperatingMode.OVERNIGHT: "Market closed (overnight)",
            OperatingMode.WEEKEND: "Market closed (weekend)",
        }
        return RiskVerdict(
            approved=False,
            violations=[
                RiskViolation(
                    rule="market_hours",
                    limit=0,
                    actual=0,
                    description=reason_map.get(mode, f"Market closed ({mode.value})"),
                )
            ],
        )

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

    # -------------------------------------------------------------------------
    # Continuous intra-trade risk monitoring (Phase 4.1)
    # -------------------------------------------------------------------------

    def monitor(
        self,
        current_regimes: dict[str, dict[str, Any]] | None = None,
        entry_regimes: dict[str, str] | None = None,
    ) -> MonitorReport:
        """
        Continuous risk check on open positions.

        Designed to run every 60s (via scheduler or AutonomousRunner). Unlike
        check() which gates new orders, monitor() evaluates existing positions
        for conditions that developed AFTER entry.

        Checks:
          1. Position size drift — stock moved, now exceeds max_position_pct
          2. Correlation spike — two positions suddenly correlated > 0.8
          3. Regime flip — entered in one regime, now in a different one
          4. Daily P&L proximity — approaching daily loss limit (>75% consumed)

        Args:
            current_regimes: {symbol: regime_dict} from latest regime detection.
            entry_regimes: {symbol: trend_regime_str} recorded at trade entry.

        Returns:
            MonitorReport with alerts and recommended actions.
        """
        current_regimes = current_regimes or {}
        entry_regimes = entry_regimes or {}
        alerts: list[MonitorAlert] = []
        snapshot = self._portfolio.get_snapshot()
        positions = self._portfolio.get_positions()
        equity = snapshot.total_equity or 100_000.0

        # -- 1. Position size drift -----------------------------------------
        for pos in positions:
            pos_pct = (abs(pos.quantity) * pos.current_price) / equity
            if pos_pct > self.limits.max_position_pct * 1.2:
                # Position has drifted >20% beyond the limit
                alerts.append(
                    MonitorAlert(
                        severity=AlertSeverity.WARNING,
                        rule="position_size_drift",
                        symbol=pos.symbol,
                        description=(
                            f"{pos.symbol} position is {pos_pct:.1%} of equity "
                            f"(limit: {self.limits.max_position_pct:.0%}) — "
                            f"drifted {((pos_pct / self.limits.max_position_pct) - 1):.0%} beyond limit"
                        ),
                        recommended_action="trim",
                        details={
                            "current_pct": round(pos_pct, 4),
                            "limit_pct": self.limits.max_position_pct,
                        },
                    )
                )

        # -- 2. Pairwise correlation spike ----------------------------------
        if len(positions) >= 2:
            corr_alerts = self._check_correlation_spikes(positions)
            alerts.extend(corr_alerts)

        # -- 3. Regime flip detection ---------------------------------------
        for pos in positions:
            sym = pos.symbol
            if sym in current_regimes and sym in entry_regimes:
                current_trend = current_regimes[sym].get("trend_regime", "unknown")
                entry_trend = entry_regimes[sym]
                if (
                    entry_trend != "unknown"
                    and current_trend != "unknown"
                    and entry_trend != current_trend
                ):
                    # Determine severity: opposite direction is critical, lateral is warning
                    opposites = {
                        ("trending_up", "trending_down"),
                        ("trending_down", "trending_up"),
                    }
                    severity = (
                        AlertSeverity.CRITICAL
                        if (entry_trend, current_trend) in opposites
                        else AlertSeverity.WARNING
                    )
                    alerts.append(
                        MonitorAlert(
                            severity=severity,
                            rule="regime_flip",
                            symbol=sym,
                            description=(
                                f"{sym} regime flipped: {entry_trend} → {current_trend} "
                                f"(entered during {entry_trend})"
                            ),
                            recommended_action=(
                                "evaluate_exit"
                                if severity == AlertSeverity.CRITICAL
                                else "reduce"
                            ),
                            details={
                                "entry_regime": entry_trend,
                                "current_regime": current_trend,
                            },
                        )
                    )

        # -- 4. Daily loss proximity ----------------------------------------
        if equity > 0:
            daily_loss_pct = abs(min(0, snapshot.daily_pnl)) / equity
            limit_pct = self.limits.daily_loss_limit_pct
            if daily_loss_pct >= limit_pct * 0.75:
                pct_consumed = daily_loss_pct / limit_pct
                alerts.append(
                    MonitorAlert(
                        severity=(
                            AlertSeverity.CRITICAL
                            if pct_consumed >= 0.90
                            else AlertSeverity.WARNING
                        ),
                        rule="daily_loss_proximity",
                        symbol="PORTFOLIO",
                        description=(
                            f"Daily loss at {daily_loss_pct:.2%} — "
                            f"{pct_consumed:.0%} of {limit_pct:.1%} limit consumed"
                        ),
                        recommended_action="halt_new_entries",
                        details={
                            "daily_loss_pct": round(daily_loss_pct, 4),
                            "limit_pct": limit_pct,
                        },
                    )
                )

        # -- Log and return -------------------------------------------------
        if alerts:
            critical_count = sum(
                1 for a in alerts if a.severity == AlertSeverity.CRITICAL
            )
            warn_count = sum(1 for a in alerts if a.severity == AlertSeverity.WARNING)
            logger.warning(
                f"[RISK MONITOR] {len(alerts)} alerts: "
                f"{critical_count} CRITICAL, {warn_count} WARNING"
            )
            for alert in alerts:
                log_fn = (
                    logger.critical
                    if alert.severity == AlertSeverity.CRITICAL
                    else logger.warning
                )
                log_fn(f"[RISK MONITOR] [{alert.rule}] {alert.description}")

        return MonitorReport(alerts=alerts, positions_checked=len(positions))

    def _check_correlation_spikes(self, positions: list) -> list[MonitorAlert]:
        """
        Check for pairwise correlation spikes among open positions.

        Uses 30-day rolling correlation of daily returns. Alerts when
        any pair exceeds 0.80 — indicates hidden concentration risk.
        """
        alerts: list[MonitorAlert] = []
        try:
            store = DataStore()
            symbols = [p.symbol for p in positions]
            returns_map: dict[str, Any] = {}

            for sym in symbols:
                df = store.load_ohlcv(sym, Timeframe.D1)
                if df is not None and len(df) >= 30:
                    returns_map[sym] = df["close"].pct_change().dropna().tail(30)

            checked_pairs: set[tuple[str, str]] = set()
            for i, sym_a in enumerate(symbols):
                for sym_b in symbols[i + 1 :]:
                    if sym_a not in returns_map or sym_b not in returns_map:
                        continue
                    pair = (min(sym_a, sym_b), max(sym_a, sym_b))
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)

                    ret_a = returns_map[sym_a]
                    ret_b = returns_map[sym_b]
                    # Align on common dates
                    common_idx = ret_a.index.intersection(ret_b.index)
                    if len(common_idx) < 20:
                        continue
                    corr = ret_a.loc[common_idx].corr(ret_b.loc[common_idx])
                    if corr > 0.80:
                        alerts.append(
                            MonitorAlert(
                                severity=AlertSeverity.WARNING,
                                rule="correlation_spike",
                                symbol=f"{sym_a}/{sym_b}",
                                description=(
                                    f"{sym_a} and {sym_b} 30-day correlation = {corr:.2f} "
                                    f"(threshold: 0.80) — hidden concentration risk"
                                ),
                                recommended_action="reduce_one",
                                details={
                                    "pair": [sym_a, sym_b],
                                    "correlation": round(corr, 3),
                                },
                            )
                        )
        except Exception as exc:
            logger.debug(f"[RISK MONITOR] correlation check failed: {exc}")

        return alerts


# =============================================================================
# Monitor data models
# =============================================================================


class AlertSeverity(str, Enum):
    """Severity level for monitor alerts."""

    WARNING = "warning"  # Informational — position approaching a limit
    CRITICAL = "critical"  # Action required — limit breached or regime reversal


@dataclass
class MonitorAlert:
    """A single alert from the continuous risk monitor."""

    severity: AlertSeverity
    rule: str  # e.g., "position_size_drift", "correlation_spike"
    symbol: str  # Affected symbol (or "PORTFOLIO" for portfolio-level)
    description: str
    recommended_action: str  # "trim", "evaluate_exit", "reduce", "halt_new_entries"
    details: dict = field(default_factory=dict)


@dataclass
class MonitorReport:
    """Result of a monitor() pass."""

    alerts: list[MonitorAlert] = field(default_factory=list)
    positions_checked: int = 0

    @property
    def has_critical(self) -> bool:
        return any(a.severity == AlertSeverity.CRITICAL for a in self.alerts)

    @property
    def action_required(self) -> bool:
        return self.has_critical


# Singleton
_risk_gate: RiskGate | None = None


def get_risk_gate(limits: RiskLimits | None = None) -> RiskGate:
    """Get the singleton RiskGate instance."""
    global _risk_gate
    if _risk_gate is None:
        _risk_gate = RiskGate(limits=limits)
    return _risk_gate
