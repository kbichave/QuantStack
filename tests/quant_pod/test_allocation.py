# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the allocation engine and conflict resolution logic (Phase 5).
"""

from __future__ import annotations

import pytest

from quant_pod.mcp.allocation import compute_allocation, resolve_conflicts


# ---------------------------------------------------------------------------
# Allocation Engine
# ---------------------------------------------------------------------------


class TestComputeAllocation:
    def test_no_strategies_returns_empty(self):
        result = compute_allocation("trending_up", 0.8, [])
        assert result["allocations"] == []
        assert result["total_allocated_pct"] == 0.0
        assert len(result["warnings"]) > 0

    def test_filters_by_status(self):
        strategies = [
            {
                "strategy_id": "s1", "name": "draft_strat", "status": "draft",
                "regime_affinity": {"trending_up": 0.9},
                "backtest_summary": {"sharpe_ratio": 2.0},
                "risk_params": {},
            },
            {
                "strategy_id": "s2", "name": "live_strat", "status": "live",
                "regime_affinity": {"trending_up": 0.8},
                "backtest_summary": {"sharpe_ratio": 1.5},
                "risk_params": {},
            },
        ]
        result = compute_allocation("trending_up", 0.8, strategies)
        ids = [a["strategy_id"] for a in result["allocations"]]
        assert "s2" in ids
        assert "s1" not in ids  # draft not eligible

    def test_filters_by_regime_affinity(self):
        strategies = [
            {
                "strategy_id": "s1", "name": "trend_strat", "status": "live",
                "regime_affinity": {"trending_up": 0.9},
                "backtest_summary": {"sharpe_ratio": 1.5},
                "risk_params": {},
            },
            {
                "strategy_id": "s2", "name": "range_strat", "status": "live",
                "regime_affinity": {"ranging": 0.9},
                "backtest_summary": {"sharpe_ratio": 1.5},
                "risk_params": {},
            },
        ]
        result = compute_allocation("trending_up", 0.8, strategies)
        ids = [a["strategy_id"] for a in result["allocations"]]
        assert "s1" in ids
        assert "s2" not in ids  # ranging doesn't match trending_up

    def test_forward_testing_capped(self):
        strategies = [
            {
                "strategy_id": "s1", "name": "ft_strat", "status": "forward_testing",
                "regime_affinity": {"trending_up": 1.0},
                "backtest_summary": {"sharpe_ratio": 3.0},
                "risk_params": {},
            },
        ]
        result = compute_allocation("trending_up", 0.9, strategies, forward_testing_cap=0.10)
        alloc = result["allocations"][0]
        assert alloc["capital_pct"] <= 0.10
        assert alloc["mode"] == "paper"

    def test_live_strategy_gets_live_mode(self):
        strategies = [
            {
                "strategy_id": "s1", "name": "live_strat", "status": "live",
                "regime_affinity": {"trending_up": 0.8},
                "backtest_summary": {"sharpe_ratio": 1.5},
                "risk_params": {},
            },
        ]
        result = compute_allocation("trending_up", 0.9, strategies)
        assert result["allocations"][0]["mode"] == "live"

    def test_low_regime_confidence_scales_down(self):
        strategies = [
            {
                "strategy_id": "s1", "name": "strat", "status": "live",
                "regime_affinity": {"trending_up": 0.8},
                "backtest_summary": {"sharpe_ratio": 1.5},
                "risk_params": {},
            },
        ]
        high_conf = compute_allocation("trending_up", 0.9, strategies)
        low_conf = compute_allocation("trending_up", 0.3, strategies)

        if high_conf["allocations"] and low_conf["allocations"]:
            assert low_conf["allocations"][0]["capital_pct"] < high_conf["allocations"][0]["capital_pct"]

    def test_total_respects_max_exposure(self):
        strategies = [
            {
                "strategy_id": f"s{i}", "name": f"strat_{i}", "status": "live",
                "regime_affinity": {"trending_up": 0.9},
                "backtest_summary": {"sharpe_ratio": 2.0},
                "risk_params": {"position_pct": 0.5},  # each wants 50%
            }
            for i in range(10)
        ]
        result = compute_allocation("trending_up", 0.9, strategies, max_gross_exposure_pct=1.5)
        assert result["total_allocated_pct"] <= 1.5

    def test_ranking_by_sharpe(self):
        strategies = [
            {
                "strategy_id": "low", "name": "low_sharpe", "status": "live",
                "regime_affinity": {"trending_up": 0.8},
                "backtest_summary": {"sharpe_ratio": 0.5},
                "risk_params": {},
            },
            {
                "strategy_id": "high", "name": "high_sharpe", "status": "live",
                "regime_affinity": {"trending_up": 0.8},
                "backtest_summary": {"sharpe_ratio": 2.5},
                "risk_params": {},
            },
        ]
        result = compute_allocation("trending_up", 0.9, strategies)
        # High sharpe should be allocated first (higher rank)
        ids = [a["strategy_id"] for a in result["allocations"]]
        assert ids[0] == "high"


# ---------------------------------------------------------------------------
# Conflict Resolution
# ---------------------------------------------------------------------------


class TestResolveConflicts:
    def test_no_conflict_single_trade(self):
        trades = [
            {"symbol": "SPY", "action": "buy", "confidence": 0.8, "strategy_id": "s1", "capital_pct": 0.05},
        ]
        result = resolve_conflicts(trades)
        assert len(result["resolved_trades"]) == 1
        assert result["conflicts_count"] == 0

    def test_same_direction_merges_conservatively(self):
        trades = [
            {"symbol": "SPY", "action": "buy", "confidence": 0.8, "strategy_id": "s1", "capital_pct": 0.10},
            {"symbol": "SPY", "action": "buy", "confidence": 0.7, "strategy_id": "s2", "capital_pct": 0.05},
        ]
        result = resolve_conflicts(trades)
        assert len(result["resolved_trades"]) == 1
        assert result["resolved_trades"][0]["capital_pct"] == 0.05  # min of the two

    def test_opposite_direction_high_vs_low_keeps_high(self):
        trades = [
            {"symbol": "SPY", "action": "buy", "confidence": 0.9, "strategy_id": "s1", "capital_pct": 0.05},
            {"symbol": "SPY", "action": "sell", "confidence": 0.4, "strategy_id": "s2", "capital_pct": 0.05},
        ]
        result = resolve_conflicts(trades)
        assert len(result["resolved_trades"]) == 1
        assert result["resolved_trades"][0]["action"] == "buy"
        assert result["conflicts_count"] == 1

    def test_opposite_direction_both_high_skips(self):
        trades = [
            {"symbol": "SPY", "action": "buy", "confidence": 0.85, "strategy_id": "s1", "capital_pct": 0.05},
            {"symbol": "SPY", "action": "sell", "confidence": 0.80, "strategy_id": "s2", "capital_pct": 0.05},
        ]
        result = resolve_conflicts(trades)
        assert len(result["resolved_trades"]) == 0
        assert result["conflicts_count"] == 1
        assert any(r["action"] == "skip" for r in result["resolutions"])

    def test_multiple_symbols_independent(self):
        trades = [
            {"symbol": "SPY", "action": "buy", "confidence": 0.8, "strategy_id": "s1", "capital_pct": 0.05},
            {"symbol": "QQQ", "action": "sell", "confidence": 0.7, "strategy_id": "s2", "capital_pct": 0.05},
        ]
        result = resolve_conflicts(trades)
        assert len(result["resolved_trades"]) == 2
        assert result["conflicts_count"] == 0

    def test_resolutions_have_reasoning(self):
        trades = [
            {"symbol": "SPY", "action": "buy", "confidence": 0.9, "strategy_id": "s1", "capital_pct": 0.05},
            {"symbol": "SPY", "action": "sell", "confidence": 0.3, "strategy_id": "s2", "capital_pct": 0.05},
        ]
        result = resolve_conflicts(trades)
        assert len(result["resolutions"]) > 0
        assert result["resolutions"][0]["reasoning"] != ""
