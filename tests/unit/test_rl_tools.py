# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for RL CrewAI tools.

Agents are never loaded from checkpoints here (no torch required).
Tests verify: graceful degradation, shadow tagging, schema validation,
snapshot registry, and factory functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from quantstack.rl.rl_tools import RLAlphaWeightTool, RLExecutionStrategyTool, RLPositionSizeTool, _PRETRADE_SNAPSHOTS, get_rl_tools, pop_pretrade_snapshot, rl_alpha_weight_tool, rl_execution_strategy_tool, rl_position_size_tool, save_pretrade_snapshot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_config(shadow=True, enabled=True):
    cfg = MagicMock()
    cfg.sizing_shadow = shadow
    cfg.execution_shadow = shadow
    cfg.meta_shadow = shadow
    cfg.enable_sizing_rl = enabled
    cfg.enable_execution_rl = enabled
    cfg.enable_meta_rl = enabled
    cfg.sizing_state_dim = 10
    cfg.execution_state_dim = 8
    cfg.alpha_selection_state_dim = 36
    cfg.sizing_checkpoint_path = Path("/nonexistent/sizing.pt")
    cfg.execution_checkpoint_path = Path("/nonexistent/execution.pt")
    cfg.meta_checkpoint_path = Path("/nonexistent/meta.pt")
    return cfg


# ---------------------------------------------------------------------------
# Snapshot store
# ---------------------------------------------------------------------------


class TestSnapshotStore:
    def setup_method(self):
        _PRETRADE_SNAPSHOTS.clear()

    def test_save_and_pop(self):
        save_pretrade_snapshot("rl_position_size", {"scale": 0.7})
        result = pop_pretrade_snapshot("rl_position_size")
        assert result is not None
        assert result["scale"] == 0.7

    def test_pop_removes_entry(self):
        save_pretrade_snapshot("rl_position_size", {"scale": 0.7})
        pop_pretrade_snapshot("rl_position_size")
        assert pop_pretrade_snapshot("rl_position_size") is None

    def test_pop_nonexistent_returns_none(self):
        assert pop_pretrade_snapshot("nonexistent") is None

    def test_overwrite_same_key(self):
        save_pretrade_snapshot("rl_position_size", {"scale": 0.3})
        save_pretrade_snapshot("rl_position_size", {"scale": 0.9})
        result = pop_pretrade_snapshot("rl_position_size")
        assert result["scale"] == 0.9


# ---------------------------------------------------------------------------
# RLPositionSizeTool
# ---------------------------------------------------------------------------


class TestRLPositionSizeTool:
    def setup_method(self):
        _PRETRADE_SNAPSHOTS.clear()

    def test_degrades_gracefully_when_no_checkpoint(self):
        tool = RLPositionSizeTool()
        with patch.object(tool, "_get_config", return_value=_mock_config()):
            result = tool._run(signal_confidence=0.7, signal_direction="LONG")
        data = json.loads(result)
        assert "scale" in data
        assert isinstance(data["scale"], float)

    def test_returns_shadow_true_when_configured(self):
        tool = RLPositionSizeTool()
        with patch.object(tool, "_get_config", return_value=_mock_config(shadow=True)):
            result = tool._run(signal_confidence=0.7, signal_direction="LONG")
        data = json.loads(result)
        assert data["shadow"] is True

    def test_disabled_returns_fallback(self):
        tool = RLPositionSizeTool()
        cfg = _mock_config()
        cfg.enable_sizing_rl = False
        with patch.object(tool, "_get_config", return_value=cfg):
            result = tool._run(signal_confidence=0.7, signal_direction="LONG")
        data = json.loads(result)
        assert "scale" in data
        assert data["scale"] == pytest.approx(0.7, abs=0.01)

    def test_scale_in_valid_range(self):
        tool = RLPositionSizeTool()
        with patch.object(tool, "_get_config", return_value=_mock_config()):
            result = tool._run(signal_confidence=0.7, signal_direction="LONG")
        data = json.loads(result)
        assert 0.0 <= data["scale"] <= 1.0

    def test_saves_pretrade_snapshot_when_agent_available(self):
        tool = RLPositionSizeTool()
        cfg = _mock_config(shadow=True)

        # Mock a functioning agent
        mock_agent = MagicMock()
        mock_action = MagicMock()
        mock_action.value = 0.6
        mock_agent.select_action.return_value = mock_action

        with patch.object(tool, "_get_config", return_value=cfg):
            with patch.object(tool, "_load_agent", return_value=mock_agent):
                tool._run(signal_confidence=0.7, signal_direction="LONG")

        snapshot = pop_pretrade_snapshot("rl_position_size")
        assert snapshot is not None
        assert snapshot["tool_name"] == "rl_position_size"
        assert "state_vector" in snapshot

    def test_result_has_reasoning(self):
        tool = RLPositionSizeTool()
        with patch.object(tool, "_get_config", return_value=_mock_config()):
            result = tool._run(signal_confidence=0.7, signal_direction="LONG")
        data = json.loads(result)
        assert "reasoning" in data


# ---------------------------------------------------------------------------
# RLExecutionStrategyTool
# ---------------------------------------------------------------------------


class TestRLExecutionStrategyTool:
    def test_degrades_gracefully_when_no_checkpoint(self):
        tool = RLExecutionStrategyTool()
        with patch.object(tool, "_get_config", return_value=_mock_config()):
            result = tool._run(symbol="SPY", quantity=100.0, urgency="normal")
        data = json.loads(result)
        assert "strategy" in data

    def test_returns_valid_strategy(self):
        tool = RLExecutionStrategyTool()
        valid_strategies = {"AGGRESSIVE", "BALANCED", "PASSIVE", "TWAP", "NO_TRADE"}
        with patch.object(tool, "_get_config", return_value=_mock_config()):
            result = tool._run(symbol="SPY", quantity=100.0, urgency="low")
        data = json.loads(result)
        assert data["strategy"] in valid_strategies

    def test_disabled_returns_fallback(self):
        tool = RLExecutionStrategyTool()
        cfg = _mock_config()
        cfg.enable_execution_rl = False
        with patch.object(tool, "_get_config", return_value=cfg):
            result = tool._run(symbol="SPY", quantity=100.0, urgency="normal")
        data = json.loads(result)
        assert "strategy" in data

    def test_shadow_flag_true_when_configured(self):
        tool = RLExecutionStrategyTool()
        with patch.object(tool, "_get_config", return_value=_mock_config(shadow=True)):
            result = tool._run(symbol="SPY", quantity=100.0, urgency="normal")
        data = json.loads(result)
        assert data.get("shadow") is True


# ---------------------------------------------------------------------------
# RLAlphaWeightTool
# ---------------------------------------------------------------------------


class TestRLAlphaWeightTool:
    def test_degrades_gracefully(self):
        tool = RLAlphaWeightTool()
        with patch.object(tool, "_get_config", return_value=_mock_config()):
            result = tool._run(
                regime="trending_up",
                competing_signals=["TREND", "MOMENTUM"],
            )
        data = json.loads(result)
        assert "selected_alpha" in data
        assert "weights" in data

    def test_disabled_returns_equal_weights(self):
        tool = RLAlphaWeightTool()
        cfg = _mock_config()
        cfg.enable_meta_rl = False
        with patch.object(tool, "_get_config", return_value=cfg):
            result = tool._run(
                regime="normal",
                competing_signals=["A", "B"],
            )
        data = json.loads(result)
        assert abs(data["weights"]["A"] - data["weights"]["B"]) < 1e-5

    def test_empty_signals(self):
        tool = RLAlphaWeightTool()
        with patch.object(tool, "_get_config", return_value=_mock_config()):
            result = tool._run(regime="normal", competing_signals=[])
        data = json.loads(result)
        assert "selected_alpha" in data


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class TestFactoryFunctions:
    def test_rl_position_size_tool_factory(self):
        tool = rl_position_size_tool()
        assert tool.name == "rl_position_size"

    def test_rl_execution_strategy_tool_factory(self):
        tool = rl_execution_strategy_tool()
        assert tool.name == "rl_execution_strategy"

    def test_rl_alpha_weight_tool_factory(self):
        tool = rl_alpha_weight_tool()
        assert tool.name == "rl_alpha_weight"

    def test_get_rl_tools_returns_list(self):
        cfg = _mock_config()
        tools = get_rl_tools(cfg)
        assert isinstance(tools, list)
        assert len(tools) == 3  # all 3 enabled

    def test_get_rl_tools_respects_disabled_flags(self):
        cfg = _mock_config()
        cfg.enable_sizing_rl = False
        cfg.enable_execution_rl = False
        cfg.enable_meta_rl = False
        tools = get_rl_tools(cfg)
        assert tools == []

    def test_get_rl_tools_partial_enabled(self):
        cfg = _mock_config()
        cfg.enable_sizing_rl = True
        cfg.enable_execution_rl = False
        cfg.enable_meta_rl = False
        tools = get_rl_tools(cfg)
        assert len(tools) == 1
        assert tools[0].name == "rl_position_size"
