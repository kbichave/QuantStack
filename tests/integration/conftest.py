"""Shared fixtures for LangGraph integration tests."""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock


CANNED_RESPONSES = {
    "safety_check": json.dumps({
        "halted": False,
        "reason": "all systems healthy",
    }),
    "daily_plan": json.dumps({
        "plan": "Focus on momentum setups in trending_up regime",
        "priorities": ["AAPL", "SPY"],
    }),
    "position_review": json.dumps([]),
    "execute_exits": json.dumps([]),
    "entry_scan": json.dumps([
        {"symbol": "AAPL", "strategy": "swing_momentum", "signal_strength": 0.8}
    ]),
    "risk_sizing": json.dumps([
        {"symbol": "AAPL", "recommended_size_pct": 5.0, "reasoning": "half-Kelly", "confidence": 0.8}
    ]),
    "portfolio_review": json.dumps([
        {"symbol": "AAPL", "decision": "APPROVED", "reason": "fits portfolio"}
    ]),
    "options_analysis": json.dumps([]),
    "execute_entries": json.dumps([]),
    "reflection": json.dumps({"reflection": "No positions closed this cycle.", "lessons": []}),
    "default": json.dumps({"result": "ok", "passed": True, "summary": "ok",
                           "domain": "swing", "symbols": ["SPY"],
                           "hypothesis": "test hypothesis",
                           "backtest_id": "bt-1", "sharpe": 1.5,
                           "experiment_id": "exp-1", "strategy_id": "strat-1",
                           "status": "paper_ready"}),
}


@pytest.fixture()
def mock_llm_response():
    """Factory that returns canned LLM responses by task name."""
    def _get(task_name: str) -> str:
        return CANNED_RESPONSES.get(task_name, CANNED_RESPONSES["default"])
    return _get


@pytest.fixture()
def mock_chat_model():
    """Mock BaseChatModel that returns canned JSON for all LLM calls."""
    from langchain_core.messages import AIMessage
    model = MagicMock()
    model.bind_tools = MagicMock(return_value=model)
    model.ainvoke = AsyncMock(return_value=AIMessage(
        content=CANNED_RESPONSES["default"]
    ))
    return model


@pytest.fixture()
def mock_checkpointer():
    """In-memory checkpointer for test isolation."""
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()
