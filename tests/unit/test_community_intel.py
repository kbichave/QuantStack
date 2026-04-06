"""Unit tests for community intel iterative search (WI-3).

Covers: agent config validation, CommunityDiscovery schema, iteration fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_type_hints

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENTS_YAML = PROJECT_ROOT / "src" / "quantstack" / "graphs" / "research" / "config" / "agents.yaml"


class TestCommunityIntelConfig:
    """Validate the community_intel agent YAML config."""

    @pytest.fixture(autouse=True)
    def _load_config(self):
        from quantstack.graphs.config import load_agent_configs
        configs = load_agent_configs(AGENTS_YAML)
        self.cfg = configs["community_intel"]

    def test_backstory_contains_iteration_instructions(self):
        backstory = self.cfg.backstory.lower()
        assert "iteration 1" in backstory or "iteration 2" in backstory
        assert "gap analysis" in backstory or "gap" in backstory
        assert "deduplicate" in backstory or "deduplication" in backstory
        assert "underrepresented" in backstory or "gap" in backstory

    def test_llm_tier_is_medium(self):
        assert self.cfg.llm_tier == "medium"

    def test_max_iterations_sufficient(self):
        assert self.cfg.max_iterations >= 15

    def test_timeout_adequate(self):
        assert self.cfg.timeout_seconds >= 300


class TestCommunityDiscoverySchema:
    """Validate the CommunityDiscovery TypedDict."""

    def test_has_all_fields(self):
        from quantstack.performance.models import CommunityDiscovery
        hints = get_type_hints(CommunityDiscovery)
        expected = {
            "title", "source", "category", "asset_class", "summary",
            "empirical_evidence", "implementation_path",
            "novelty_vs_registry", "iteration_found",
        }
        assert expected == set(hints.keys())

    def test_iteration_found_accepts_valid_values(self):
        from quantstack.performance.models import CommunityDiscovery
        for i in (1, 2, 3):
            d: CommunityDiscovery = {
                "title": "test",
                "source": "reddit",
                "category": "strategy",
                "asset_class": "equity",
                "summary": "test summary",
                "empirical_evidence": "backtest shows 1.5 Sharpe",
                "implementation_path": "use existing tools",
                "novelty_vs_registry": "new approach",
                "iteration_found": i,
            }
            assert d["iteration_found"] == i
