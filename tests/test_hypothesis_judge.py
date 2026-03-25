# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for HypothesisJudge — inner loop gate before backtests."""

from __future__ import annotations

import pytest

from quantstack.autonomous.judge import HypothesisJudge, JudgeVerdict
from quantstack.db import pg_conn, run_migrations


@pytest.fixture
def conn():
    with pg_conn() as c:
        run_migrations(c)
        yield c


@pytest.fixture
def judge(conn):
    return HypothesisJudge(conn)


class TestLookaheadBias:
    def test_future_feature_detected(self, judge):
        hypothesis = {
            "name": "test_strategy",
            "features": ["rsi_14", "next_day_return", "adx_14"],
        }
        verdict = judge.review(hypothesis)
        assert not verdict.approved
        assert any("lookahead" in f for f in verdict.flags)

    def test_negative_shift_in_code(self, judge):
        hypothesis = {"name": "test", "features": ["rsi_14"]}
        code = "df['signal'] = df['close'].shift(-1) > df['close']"
        verdict = judge.review(hypothesis, code=code)
        assert any("negative_shift" in f for f in verdict.flags)

    def test_clean_features_pass(self, judge):
        hypothesis = {
            "name": "test",
            "features": ["rsi_14", "macd_histogram", "bb_width", "adx_14"],
        }
        verdict = judge.review(hypothesis)
        assert not any("lookahead" in f for f in verdict.flags)


class TestKnownFailures:
    def test_momentum_in_ranging(self, judge):
        hypothesis = {
            "name": "momentum_breakout",
            "description": "momentum strategy for ranging markets",
            "regime_target": "ranging",
        }
        verdict = judge.review(hypothesis)
        assert any("known_failure" in f for f in verdict.flags)

    def test_mean_reversion_in_trending(self, judge):
        hypothesis = {
            "name": "mean_reversion_rsi",
            "description": "Buy oversold, sell overbought",
            "regime_target": "trending_up",
        }
        verdict = judge.review(hypothesis)
        assert any("known_failure" in f for f in verdict.flags)

    def test_valid_combination_passes(self, judge):
        hypothesis = {
            "name": "momentum_trend",
            "description": "trend following",
            "regime_target": "trending_up",
        }
        verdict = judge.review(hypothesis)
        assert not any("known_failure" in f for f in verdict.flags)


class TestDataSnooping:
    def test_too_many_params(self, judge):
        hypothesis = {
            "name": "overfit_strategy",
            "parameters": {f"param_{i}": i for i in range(50)},
            "data_points": 100,
        }
        verdict = judge.review(hypothesis)
        assert any("data_snooping" in f for f in verdict.flags)

    def test_reasonable_params_pass(self, judge):
        hypothesis = {
            "name": "simple_strategy",
            "parameters": {"rsi_period": 14, "threshold": 30},
            "data_points": 1000,
        }
        verdict = judge.review(hypothesis)
        assert not any("data_snooping:ratio" in f for f in verdict.flags)

    def test_too_many_entry_rules(self, judge):
        hypothesis = {
            "name": "complex_strategy",
            "entry_rules": [{"rule": f"r{i}"} for i in range(10)],
        }
        verdict = judge.review(hypothesis)
        assert any("too_many_entry_rules" in f for f in verdict.flags)


class TestCleanHypothesis:
    def test_approved(self, judge):
        hypothesis = {
            "name": "regime_momentum_v2",
            "description": "Follow momentum in trending markets",
            "features": ["rsi_14", "macd_histogram", "adx_14"],
            "regime_target": "trending_up",
            "parameters": {"rsi_period": 14, "adx_threshold": 25},
            "data_points": 2000,
            "entry_rules": [{"indicator": "rsi_14", "condition": "below", "value": 35}],
        }
        verdict = judge.review(hypothesis)
        assert verdict.approved
        assert verdict.score >= 0.5


class TestKnowledgeUpdate:
    def test_failed_backtest_added(self, judge):
        initial_size = len(judge._knowledge_base)
        judge.update_knowledge("bad_strategy", {
            "sharpe": -0.5, "win_rate": 0.35, "n_trades": 50,
        })
        assert len(judge._knowledge_base) == initial_size + 1

    def test_good_backtest_not_added(self, judge):
        initial_size = len(judge._knowledge_base)
        judge.update_knowledge("good_strategy", {
            "sharpe": 1.2, "win_rate": 0.55, "n_trades": 50,
        })
        assert len(judge._knowledge_base) == initial_size

    def test_few_trades_not_added(self, judge):
        initial_size = len(judge._knowledge_base)
        judge.update_knowledge("sparse_strategy", {
            "sharpe": -1.0, "win_rate": 0.20, "n_trades": 3,
        })
        assert len(judge._knowledge_base) == initial_size


class TestPersistence:
    def test_verdict_persisted(self, judge, conn):
        hypothesis = {"name": "test_persist"}
        judge.review(hypothesis)
        count = conn.execute("SELECT COUNT(*) FROM judge_verdicts").fetchone()[0]
        assert count == 1
