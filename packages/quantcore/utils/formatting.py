"""
Formatting helpers for console output.
"""


def print_header(title: str) -> None:
    """Print a prominent header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str) -> None:
    """Print a section divider."""
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}\n")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"✓ {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"✗ {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"⚠ {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"ℹ {message}")


def print_money(label: str, amount: float, is_profit: bool = None) -> None:
    """Print a monetary value with formatting."""
    if is_profit is None:
        is_profit = amount >= 0
    sign = "+" if amount >= 0 else ""
    emoji = "📈" if is_profit else "📉"
    print(f"{emoji} {label}: {sign}${amount:,.2f}")
