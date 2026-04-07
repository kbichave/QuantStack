"""3-window backtest patience protocol."""

from quantstack.core.backtesting.patience import PatienceConfig, WindowResult, evaluate_patience


def test_rejected_only_when_all_three_windows_fail():
    """A hypothesis is rejected only when it fails gates in ALL 3 windows."""
    results = [
        WindowResult(window_name="full", passed=False, sharpe=0.2, max_drawdown=-0.3, ic=0.01),
        WindowResult(window_name="recent", passed=False, sharpe=0.1, max_drawdown=-0.4, ic=0.005),
        WindowResult(window_name="stressed", passed=False, sharpe=-0.1, max_drawdown=-0.5, ic=-0.01),
    ]
    assert evaluate_patience(results) == "rejected"


def test_provisional_when_two_of_three_pass():
    """A hypothesis is marked 'provisional' when exactly 2 of 3 windows pass."""
    results = [
        WindowResult(window_name="full", passed=True, sharpe=1.0, max_drawdown=-0.1, ic=0.05),
        WindowResult(window_name="recent", passed=True, sharpe=0.8, max_drawdown=-0.15, ic=0.04),
        WindowResult(window_name="stressed", passed=False, sharpe=-0.2, max_drawdown=-0.4, ic=-0.01),
    ]
    assert evaluate_patience(results) == "provisional"


def test_fully_accepted_when_three_of_three_pass():
    """A hypothesis is fully accepted when all 3 windows pass."""
    results = [
        WindowResult(window_name="full", passed=True, sharpe=1.2, max_drawdown=-0.08, ic=0.06),
        WindowResult(window_name="recent", passed=True, sharpe=1.0, max_drawdown=-0.1, ic=0.05),
        WindowResult(window_name="stressed", passed=True, sharpe=0.5, max_drawdown=-0.2, ic=0.03),
    ]
    assert evaluate_patience(results) == "accepted"


def test_three_windows_are_full_recent_stressed():
    """The 3 windows are: full historical period, recent 12 months,
    and a configurable stressed period (default: 2020-03 to 2020-06)."""
    config = PatienceConfig()
    assert config.full_start == "2020-01-01"
    assert config.recent_months == 12
    assert config.stressed_start == "2020-03-01"
    assert config.stressed_end == "2020-06-30"


def test_stressed_period_configurable():
    """The stressed period start/end dates are configurable, not hardcoded."""
    config = PatienceConfig(stressed_start="2022-06-01", stressed_end="2022-09-30")
    assert config.stressed_start == "2022-06-01"
    assert config.stressed_end == "2022-09-30"


def test_single_pass_is_rejected():
    """Only 1 of 3 windows passing should result in rejection."""
    results = [
        WindowResult(window_name="full", passed=True, sharpe=0.8, max_drawdown=-0.15, ic=0.04),
        WindowResult(window_name="recent", passed=False, sharpe=0.1, max_drawdown=-0.3, ic=0.005),
        WindowResult(window_name="stressed", passed=False, sharpe=-0.1, max_drawdown=-0.5, ic=-0.01),
    ]
    assert evaluate_patience(results) == "rejected"
