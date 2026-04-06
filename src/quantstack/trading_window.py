# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Trading Window configuration.

Defines the allowed instrument types and time horizons for research and trading
contexts. Each TradingWindow maps to concrete constraints: DTE bounds for options,
hold-period bounds for equity, and compatible HoldingType sets.

Env vars:
  RESEARCH_WINDOW — comma-separated TradingWindow values (default: all)
  TRADING_WINDOW  — comma-separated TradingWindow values (default: all)

Examples:
  TRADING_WINDOW=options_short_term
  TRADING_WINDOW=options_weekly,options_monthly,equity_swing
  RESEARCH_WINDOW=all
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from quantstack.holding_period import HoldingType


# ---------------------------------------------------------------------------
# Instrument type
# ---------------------------------------------------------------------------


class InstrumentType(str, Enum):
    """Instrument classification for trade gating."""

    EQUITY = "equity"
    OPTIONS = "options"


# ---------------------------------------------------------------------------
# Trading window enum — 14 leaves + 6 composites
# ---------------------------------------------------------------------------


class TradingWindow(str, Enum):
    """
    Allowed trading windows.

    Leaf values define concrete DTE or hold-period bounds.
    Composite values expand to a set of leaves for convenience.
    """

    # --- Options leaves ---
    OPTIONS_0DTE = "options_0dte"
    OPTIONS_WEEKLY = "options_weekly"
    OPTIONS_BIWEEKLY = "options_biweekly"
    OPTIONS_MONTHLY = "options_monthly"
    OPTIONS_QUARTERLY = "options_quarterly"
    OPTIONS_6_MONTH = "options_6_month"
    OPTIONS_LEAPS = "options_leaps"
    OPTIONS_ALL = "options_all"

    # --- Equity leaves ---
    EQUITY_SCALP = "equity_scalp"
    EQUITY_DAY_TRADE = "equity_day_trade"
    EQUITY_SWING = "equity_swing"
    EQUITY_POSITION = "equity_position"
    EQUITY_INVESTMENT = "equity_investment"
    EQUITY_ALL = "equity_all"

    # --- Composite shortcuts ---
    ALL = "all"
    OPTIONS_SHORT_TERM = "options_short_term"
    OPTIONS_MEDIUM_TERM = "options_medium_term"
    OPTIONS_LONG_TERM = "options_long_term"
    EQUITY_SHORT_TERM = "equity_short_term"
    EQUITY_LONG_TERM = "equity_long_term"


# ---------------------------------------------------------------------------
# Window specification — metadata per leaf
# ---------------------------------------------------------------------------

_ALL_HOLDING_TYPES = frozenset(HoldingType)


@dataclass(frozen=True)
class WindowSpec:
    """Concrete constraints for a single leaf TradingWindow."""

    window: TradingWindow
    instrument_type: InstrumentType
    dte_min: int | None = None  # Options only
    dte_max: int | None = None  # Options only
    hold_days_min: int | None = None  # Equity only
    hold_days_max: int | None = None  # Equity only
    compatible_holding_types: frozenset[HoldingType] = _ALL_HOLDING_TYPES


# ---------------------------------------------------------------------------
# Registry — one WindowSpec per leaf
# ---------------------------------------------------------------------------

WINDOW_SPECS: dict[TradingWindow, WindowSpec] = {
    # Options
    TradingWindow.OPTIONS_0DTE: WindowSpec(
        window=TradingWindow.OPTIONS_0DTE,
        instrument_type=InstrumentType.OPTIONS,
        dte_min=0,
        dte_max=0,
        compatible_holding_types=frozenset({HoldingType.INTRADAY}),
    ),
    TradingWindow.OPTIONS_WEEKLY: WindowSpec(
        window=TradingWindow.OPTIONS_WEEKLY,
        instrument_type=InstrumentType.OPTIONS,
        dte_min=1,
        dte_max=7,
        compatible_holding_types=frozenset(
            {HoldingType.INTRADAY, HoldingType.SHORT_SWING}
        ),
    ),
    TradingWindow.OPTIONS_BIWEEKLY: WindowSpec(
        window=TradingWindow.OPTIONS_BIWEEKLY,
        instrument_type=InstrumentType.OPTIONS,
        dte_min=7,
        dte_max=14,
        compatible_holding_types=frozenset({HoldingType.SHORT_SWING}),
    ),
    TradingWindow.OPTIONS_MONTHLY: WindowSpec(
        window=TradingWindow.OPTIONS_MONTHLY,
        instrument_type=InstrumentType.OPTIONS,
        dte_min=14,
        dte_max=45,
        compatible_holding_types=frozenset(
            {HoldingType.SHORT_SWING, HoldingType.SWING}
        ),
    ),
    TradingWindow.OPTIONS_QUARTERLY: WindowSpec(
        window=TradingWindow.OPTIONS_QUARTERLY,
        instrument_type=InstrumentType.OPTIONS,
        dte_min=45,
        dte_max=90,
        compatible_holding_types=frozenset(
            {HoldingType.SWING, HoldingType.POSITION}
        ),
    ),
    TradingWindow.OPTIONS_6_MONTH: WindowSpec(
        window=TradingWindow.OPTIONS_6_MONTH,
        instrument_type=InstrumentType.OPTIONS,
        dte_min=90,
        dte_max=180,
        compatible_holding_types=frozenset({HoldingType.POSITION}),
    ),
    TradingWindow.OPTIONS_LEAPS: WindowSpec(
        window=TradingWindow.OPTIONS_LEAPS,
        instrument_type=InstrumentType.OPTIONS,
        dte_min=180,
        dte_max=730,
        compatible_holding_types=frozenset({HoldingType.POSITION}),
    ),
    TradingWindow.OPTIONS_ALL: WindowSpec(
        window=TradingWindow.OPTIONS_ALL,
        instrument_type=InstrumentType.OPTIONS,
        dte_min=0,
        dte_max=730,
        compatible_holding_types=_ALL_HOLDING_TYPES,
    ),
    # Equity
    TradingWindow.EQUITY_SCALP: WindowSpec(
        window=TradingWindow.EQUITY_SCALP,
        instrument_type=InstrumentType.EQUITY,
        hold_days_min=0,
        hold_days_max=0,
        compatible_holding_types=frozenset({HoldingType.INTRADAY}),
    ),
    TradingWindow.EQUITY_DAY_TRADE: WindowSpec(
        window=TradingWindow.EQUITY_DAY_TRADE,
        instrument_type=InstrumentType.EQUITY,
        hold_days_min=0,
        hold_days_max=1,
        compatible_holding_types=frozenset({HoldingType.INTRADAY}),
    ),
    TradingWindow.EQUITY_SWING: WindowSpec(
        window=TradingWindow.EQUITY_SWING,
        instrument_type=InstrumentType.EQUITY,
        hold_days_min=1,
        hold_days_max=10,
        compatible_holding_types=frozenset(
            {HoldingType.SHORT_SWING, HoldingType.SWING}
        ),
    ),
    TradingWindow.EQUITY_POSITION: WindowSpec(
        window=TradingWindow.EQUITY_POSITION,
        instrument_type=InstrumentType.EQUITY,
        hold_days_min=10,
        hold_days_max=90,
        compatible_holding_types=frozenset(
            {HoldingType.SWING, HoldingType.POSITION}
        ),
    ),
    TradingWindow.EQUITY_INVESTMENT: WindowSpec(
        window=TradingWindow.EQUITY_INVESTMENT,
        instrument_type=InstrumentType.EQUITY,
        hold_days_min=90,
        hold_days_max=999,
        compatible_holding_types=frozenset({HoldingType.POSITION}),
    ),
    TradingWindow.EQUITY_ALL: WindowSpec(
        window=TradingWindow.EQUITY_ALL,
        instrument_type=InstrumentType.EQUITY,
        hold_days_min=0,
        hold_days_max=999,
        compatible_holding_types=_ALL_HOLDING_TYPES,
    ),
}

# All leaf windows (those with a WindowSpec entry)
_LEAF_WINDOWS: frozenset[TradingWindow] = frozenset(WINDOW_SPECS)

# Options leaves only
_OPTIONS_LEAVES: frozenset[TradingWindow] = frozenset(
    w for w, spec in WINDOW_SPECS.items()
    if spec.instrument_type == InstrumentType.OPTIONS
)

# Equity leaves only
_EQUITY_LEAVES: frozenset[TradingWindow] = frozenset(
    w for w, spec in WINDOW_SPECS.items()
    if spec.instrument_type == InstrumentType.EQUITY
)

# ---------------------------------------------------------------------------
# Composite expansion map
# ---------------------------------------------------------------------------

COMPOSITE_EXPANSIONS: dict[TradingWindow, frozenset[TradingWindow]] = {
    TradingWindow.ALL: _LEAF_WINDOWS,
    TradingWindow.OPTIONS_SHORT_TERM: frozenset({
        TradingWindow.OPTIONS_0DTE,
        TradingWindow.OPTIONS_WEEKLY,
        TradingWindow.OPTIONS_BIWEEKLY,
    }),
    TradingWindow.OPTIONS_MEDIUM_TERM: frozenset({
        TradingWindow.OPTIONS_MONTHLY,
        TradingWindow.OPTIONS_QUARTERLY,
    }),
    TradingWindow.OPTIONS_LONG_TERM: frozenset({
        TradingWindow.OPTIONS_6_MONTH,
        TradingWindow.OPTIONS_LEAPS,
    }),
    TradingWindow.EQUITY_SHORT_TERM: frozenset({
        TradingWindow.EQUITY_SCALP,
        TradingWindow.EQUITY_DAY_TRADE,
        TradingWindow.EQUITY_SWING,
    }),
    TradingWindow.EQUITY_LONG_TERM: frozenset({
        TradingWindow.EQUITY_POSITION,
        TradingWindow.EQUITY_INVESTMENT,
    }),
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def expand_windows(windows: set[TradingWindow]) -> set[TradingWindow]:
    """Expand composite windows to their leaf constituents.

    Leaf values pass through unchanged. Composites are replaced by
    the set of leaves they contain.
    """
    result: set[TradingWindow] = set()
    for w in windows:
        if w in COMPOSITE_EXPANSIONS:
            result.update(COMPOSITE_EXPANSIONS[w])
        elif w in _LEAF_WINDOWS:
            result.add(w)
        else:
            raise ValueError(
                f"Unknown TradingWindow '{w.value}' is neither leaf nor composite"
            )
    return result


def parse_window_env(raw: str) -> set[TradingWindow]:
    """Parse a comma-separated env var into an expanded set of leaf windows.

    Case-insensitive. Empty/unset string defaults to ALL (all leaves).

    Raises:
        ValueError: If any token is not a valid TradingWindow name.
    """
    valid_names = {tw.value: tw for tw in TradingWindow}

    tokens = [t.strip().lower() for t in raw.split(",") if t.strip()]
    if not tokens:
        return expand_windows({TradingWindow.ALL})

    parsed: set[TradingWindow] = set()
    for token in tokens:
        if token not in valid_names:
            raise ValueError(
                f"Invalid trading window: '{token}'. "
                f"Valid values: {', '.join(sorted(valid_names))}"
            )
        parsed.add(valid_names[token])

    return expand_windows(parsed)


def is_trade_allowed(
    instrument_type: InstrumentType | str,
    allowed_windows: set[TradingWindow],
    *,
    dte: int | None = None,
    hold_days: int | None = None,
) -> bool:
    """Check whether a trade's parameters fall within any allowed leaf window.

    Args:
        instrument_type: The instrument being traded.
        allowed_windows: Set of *expanded* leaf windows (call expand_windows first
            if the set may contain composites).
        dte: Days to expiration (required for options).
        hold_days: Expected holding period in days (optional for equity).

    Returns:
        True if the trade is permitted by at least one allowed window.
    """
    if isinstance(instrument_type, str):
        instrument_type = InstrumentType(instrument_type)

    for w in allowed_windows:
        spec = WINDOW_SPECS.get(w)
        if spec is None:
            continue  # composite that slipped through — skip

        if spec.instrument_type != instrument_type:
            continue

        if instrument_type == InstrumentType.OPTIONS:
            if dte is None:
                continue  # can't validate without DTE
            if spec.dte_min is not None and dte < spec.dte_min:
                continue
            if spec.dte_max is not None and dte > spec.dte_max:
                continue
            return True

        # Equity — hold_days is optional; if not provided, allow
        if hold_days is not None:
            if spec.hold_days_min is not None and hold_days < spec.hold_days_min:
                continue
            if spec.hold_days_max is not None and hold_days > spec.hold_days_max:
                continue
        return True

    return False


def get_dte_bounds(windows: set[TradingWindow]) -> tuple[int, int] | None:
    """Compute the effective min/max DTE across all options windows.

    Returns None if no options windows are in the set.
    """
    options_specs = [
        WINDOW_SPECS[w]
        for w in windows
        if w in WINDOW_SPECS and WINDOW_SPECS[w].instrument_type == InstrumentType.OPTIONS
    ]
    if not options_specs:
        return None

    min_dte = min(s.dte_min for s in options_specs if s.dte_min is not None)
    max_dte = max(s.dte_max for s in options_specs if s.dte_max is not None)
    return (min_dte, max_dte)


def get_hold_days_bounds(windows: set[TradingWindow]) -> tuple[int, int] | None:
    """Compute the effective min/max hold days across all equity windows.

    Returns None if no equity windows are in the set.
    """
    equity_specs = [
        WINDOW_SPECS[w]
        for w in windows
        if w in WINDOW_SPECS and WINDOW_SPECS[w].instrument_type == InstrumentType.EQUITY
    ]
    if not equity_specs:
        return None

    min_days = min(s.hold_days_min for s in equity_specs if s.hold_days_min is not None)
    max_days = max(s.hold_days_max for s in equity_specs if s.hold_days_max is not None)
    return (min_days, max_days)


def allowed_holding_types(windows: set[TradingWindow]) -> set[HoldingType]:
    """Union of compatible HoldingTypes across all windows in the set."""
    result: set[HoldingType] = set()
    for w in windows:
        spec = WINDOW_SPECS.get(w)
        if spec is not None:
            result.update(spec.compatible_holding_types)
    return result


def allowed_instrument_types(windows: set[TradingWindow]) -> set[InstrumentType]:
    """Set of instrument types permitted by the given windows."""
    result: set[InstrumentType] = set()
    for w in windows:
        spec = WINDOW_SPECS.get(w)
        if spec is not None:
            result.add(spec.instrument_type)
    return result
