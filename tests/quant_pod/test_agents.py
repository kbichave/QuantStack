# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for agent classes."""

import pytest
from quant_pod.agents.regime_detector import RegimeDetectorAgent


class TestRegimeDetectorAgent:
    """Test regime detector agent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = RegimeDetectorAgent()

        assert "SPY" in agent.symbols

    def test_custom_symbols(self):
        """Test with custom symbols."""
        symbols = ["AAPL", "TSLA"]
        agent = RegimeDetectorAgent(symbols=symbols)

        assert agent.symbols == symbols

    def test_thresholds(self):
        """Test threshold values."""
        agent = RegimeDetectorAgent()

        assert agent.ADX_TRENDING_THRESHOLD == 25
        assert agent.VOL_LOW_PCT == 25
        assert agent.VOL_HIGH_PCT == 75


class TestSchemaReexports:
    """Test that schemas are properly re-exported."""

    def test_trade_decision_importable(self):
        """Test TradeDecision can be imported from agents."""
        from quant_pod.agents import TradeDecision

        assert TradeDecision is not None

    def test_daily_brief_importable(self):
        """Test DailyBrief can be imported from agents."""
        from quant_pod.agents import DailyBrief

        assert DailyBrief is not None

    def test_analysis_note_importable(self):
        """Test AnalysisNote can be imported from agents."""
        from quant_pod.agents import AnalysisNote

        assert AnalysisNote is not None


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
