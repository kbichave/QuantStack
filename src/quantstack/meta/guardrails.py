"""Safety guardrails for meta-agent self-modification.

Meta agents can modify prompts, thresholds, and tool bindings, but certain
files are absolutely off-limits.  The risk gate, kill switch, and core DB
layer are load-bearing safety infrastructure --- a bad auto-patch there could
cause uncontrolled capital loss.

Every proposed meta change must pass ``validate_meta_change`` before being
applied.
"""

from __future__ import annotations

import fnmatch

PROTECTED_FILES: frozenset[str] = frozenset(
    {
        "src/quantstack/execution/risk_gate.py",
        "src/quantstack/execution/kill_switch.py",
        "src/quantstack/db.py",
    }
)

_PROTECTED_PATTERNS: tuple[str, ...] = ("src/quantstack/execution/*",)

# Commit prefix convention for meta-agent changes.
META_COMMIT_PREFIX = "meta:"


def is_protected(file_path: str) -> bool:
    """Return True if *file_path* must not be modified by meta agents."""
    if file_path in PROTECTED_FILES:
        return True
    return any(fnmatch.fnmatch(file_path, pat) for pat in _PROTECTED_PATTERNS)


def check_sharpe_regression(
    before_sharpe: float,
    after_sharpe: float,
    threshold: float = 0.10,
) -> bool:
    """Return True if the Sharpe ratio declined by more than *threshold* fraction.

    A True result means the change should be **reverted**.
    """
    if before_sharpe <= 0:
        # Cannot compute meaningful relative decline when baseline is <= 0.
        return after_sharpe < before_sharpe
    decline = (before_sharpe - after_sharpe) / before_sharpe
    return decline > threshold


def validate_meta_change(
    changed_files: list[str],
    test_result: bool,
    sharpe_before: float | None,
    sharpe_after: float | None,
) -> tuple[bool, str]:
    """Gate-check a proposed meta-agent modification.

    Returns ``(approved, reason)``.  When *approved* is False the change must
    not be applied.
    """
    for path in changed_files:
        if is_protected(path):
            return False, f"Protected file cannot be modified: {path}"

    if not test_result:
        return False, "Test suite did not pass after proposed change"

    if sharpe_before is not None and sharpe_after is not None:
        if check_sharpe_regression(sharpe_before, sharpe_after):
            return False, (
                f"Sharpe regression: {sharpe_before:.4f} -> {sharpe_after:.4f} "
                f"(>{10}% decline)"
            )

    return True, "Change approved"
