"""Budget tracking on graph state and exhaustion routing."""

from quantstack.graphs.state import ResearchState, TradingState


def test_research_state_default_budget():
    """ResearchState initializes with token_budget_remaining=50_000 and cost_budget_remaining=0.50."""
    state = ResearchState()
    assert state.token_budget_remaining == 50_000
    assert state.cost_budget_remaining == 0.50
    assert state.tokens_consumed == 0
    assert state.cost_consumed == 0.0


def test_trading_state_default_budget():
    """TradingState initializes with token_budget_remaining=30_000 and cost_budget_remaining=0.20."""
    state = TradingState()
    assert state.token_budget_remaining == 30_000
    assert state.cost_budget_remaining == 0.20
    assert state.tokens_consumed == 0
    assert state.cost_consumed == 0.0


def test_budget_check_continue_when_remaining_exceeds_estimate():
    """budget_check returns 'continue' when remaining > estimated_next_cost."""
    from quantstack.graphs.research.nodes import budget_check

    state = ResearchState(token_budget_remaining=40_000, cost_budget_remaining=0.40)
    assert budget_check(state) == "continue"


def test_budget_check_synthesize_when_remaining_below_estimate():
    """budget_check returns 'synthesize' when remaining < estimated_next_cost."""
    from quantstack.graphs.research.nodes import budget_check

    state = ResearchState(token_budget_remaining=2_000, cost_budget_remaining=0.001)
    assert budget_check(state) == "synthesize"


def test_budget_check_synthesize_when_remaining_zero():
    """budget_check returns 'synthesize' when remaining == 0."""
    from quantstack.graphs.research.nodes import budget_check

    state = ResearchState(token_budget_remaining=0, cost_budget_remaining=0.0)
    assert budget_check(state) == "synthesize"


def test_synthesize_partial_results_produces_summary():
    """synthesize_partial_results returns a dict with 'decisions' key."""
    from quantstack.graphs.research.nodes import synthesize_partial_results

    state = ResearchState(
        hypothesis="Test hypothesis",
        tokens_consumed=30_000,
        cost_consumed=0.35,
    )
    result = synthesize_partial_results(state)
    assert "decisions" in result
    assert len(result["decisions"]) > 0
    assert "budget_exhausted" in result["decisions"][0].get("reason", "")


def test_budget_fields_compatible_with_extra_forbid():
    """Constructing ResearchState with the new budget fields does not raise."""
    state = ResearchState(
        token_budget_remaining=100_000,
        cost_budget_remaining=1.0,
        tokens_consumed=5000,
        cost_consumed=0.05,
    )
    assert state.token_budget_remaining == 100_000
