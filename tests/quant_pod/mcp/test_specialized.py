# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for specialized MCP tool modules:
  - nlp.py (analyze_text_sentiment)
  - intraday.py (get_intraday_status, get_tca_report, get_algo_recommendation)
  - capitulation.py (get_capitulation_score)
  - institutional_accumulation.py (get_institutional_accumulation)
  - cross_domain.py (get_cross_domain_intel)
  - options_execution.py (execute_options_trade)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tests.quantstack.mcp.conftest import _fn, synthetic_ohlcv


# ===========================================================================
# nlp.py — analyze_text_sentiment
# ===========================================================================


class TestAnalyzeTextSentiment:

    @pytest.mark.asyncio
    async def test_empty_text(self):
        from quantstack.mcp.tools.nlp import analyze_text_sentiment

        result = await _fn(analyze_text_sentiment)(text="", method="groq")
        assert "error" in result
        assert result["sentiment"] == "neutral"
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_whitespace_only(self):
        from quantstack.mcp.tools.nlp import analyze_text_sentiment

        result = await _fn(analyze_text_sentiment)(text="   ", method="groq")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_method(self):
        from quantstack.mcp.tools.nlp import analyze_text_sentiment

        result = await _fn(analyze_text_sentiment)(text="some text", method="invalid")
        assert "error" in result
        assert "invalid" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_groq_happy_path(self):
        """Groq backend returns structured sentiment via mocked LiteLLM."""
        from quantstack.mcp.tools.nlp import analyze_text_sentiment

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "sentiment": "bullish",
            "confidence": 0.85,
            "dimensions": {
                "revenue_outlook": "positive",
                "guidance": "raised",
                "management_tone": "confident",
            },
            "key_phrases": ["strong revenue growth", "guidance raised"],
        })

        with patch("quantstack.mcp.tools.nlp.litellm.completion", return_value=mock_response):
            result = await _fn(analyze_text_sentiment)(
                text="Revenue exceeded expectations by 20%. Management raised full-year guidance.",
                method="groq",
            )

        assert result["sentiment"] == "bullish"
        assert result["confidence"] == 0.85
        assert result["method"] == "groq"
        assert result["dimensions"]["guidance"] == "raised"

    @pytest.mark.asyncio
    async def test_groq_json_parse_failure(self):
        """When Groq returns non-JSON, returns graceful error."""
        from quantstack.mcp.tools.nlp import analyze_text_sentiment

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is not JSON"

        with patch("quantstack.mcp.tools.nlp.litellm.completion", return_value=mock_response):
            result = await _fn(analyze_text_sentiment)(
                text="Some financial text",
                method="groq",
            )

        assert "error" in result
        assert result["sentiment"] == "neutral"
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_groq_api_failure(self):
        """When Groq API raises, returns error dict."""
        from quantstack.mcp.tools.nlp import analyze_text_sentiment

        with patch("quantstack.mcp.tools.nlp.litellm.completion", side_effect=Exception("API down")):
            result = await _fn(analyze_text_sentiment)(
                text="Some financial text",
                method="groq",
            )

        assert "error" in result
        assert result["sentiment"] == "neutral"

    @pytest.mark.asyncio
    async def test_groq_markdown_fence_stripping(self):
        """Groq response wrapped in markdown fences is parsed correctly."""
        from quantstack.mcp.tools.nlp import _parse_groq_response

        raw = '```json\n{"sentiment": "bearish", "confidence": 0.7, "dimensions": {}, "key_phrases": []}\n```'
        result = _parse_groq_response(raw)
        assert result["sentiment"] == "bearish"
        assert result["confidence"] == 0.7

    @pytest.mark.asyncio
    async def test_groq_invalid_sentiment_normalized(self):
        """Invalid sentiment values are normalized to 'neutral'."""
        from quantstack.mcp.tools.nlp import _parse_groq_response

        raw = json.dumps({
            "sentiment": "very_bullish",
            "confidence": 0.9,
            "dimensions": {},
            "key_phrases": [],
        })
        result = _parse_groq_response(raw)
        assert result["sentiment"] == "neutral"

    @pytest.mark.asyncio
    async def test_groq_confidence_clamped(self):
        """Confidence values outside 0-1 are clamped."""
        from quantstack.mcp.tools.nlp import _parse_groq_response

        raw = json.dumps({
            "sentiment": "bullish",
            "confidence": 1.5,
            "dimensions": {},
            "key_phrases": [],
        })
        result = _parse_groq_response(raw)
        assert result["confidence"] == 1.0


# ===========================================================================
# intraday.py — get_intraday_status
# ===========================================================================


class TestGetIntradayStatus:

    @pytest.mark.asyncio
    async def test_no_active_loop(self):
        """When no loop is running, returns dormant status."""
        from quantstack.mcp.tools.intraday import get_intraday_status

        with patch("quantstack.mcp.tools.intraday._get_active_loop", return_value=None):
            result = await _fn(get_intraday_status)()

        assert result["success"] is True
        assert result["running"] is False
        assert result["positions_held"] == 0

    @pytest.mark.asyncio
    async def test_active_loop(self):
        """When a loop is running, returns live status."""
        from quantstack.mcp.tools.intraday import get_intraday_status

        mock_loop = MagicMock()
        mock_loop._bar_count = 42
        mock_loop._symbols = ["SPY", "QQQ"]

        mock_pm = MagicMock()
        mock_pm._position_meta = {"SPY": {}}
        mock_pm.intraday_pnl = 150.0
        mock_pm.trades_today = 3
        mock_pm.is_flattened = False

        with (
            patch("quantstack.mcp.tools.intraday._get_active_loop", return_value=mock_loop),
            patch("quantstack.mcp.tools.intraday._find_position_manager", return_value=mock_pm),
        ):
            result = await _fn(get_intraday_status)()

        assert result["success"] is True
        assert result["running"] is True
        assert result["bars_processed"] == 42
        assert result["positions_held"] == 1
        assert result["realized_pnl"] == 150.0


# ===========================================================================
# intraday.py — get_tca_report
# ===========================================================================


class TestGetTcaReport:

    @pytest.mark.asyncio
    async def test_happy_path(self):
        """TCAStore returns aggregate stats."""
        from quantstack.mcp.tools.intraday import get_tca_report

        mock_stats = {
            "avg_slippage_bps": 2.5,
            "worst_fills": [],
            "trade_count": 10,
        }

        with patch("quantstack.mcp.tools.intraday.TCAStore") as MockTCA:
            mock_store = MagicMock()
            mock_store.get_aggregate_stats.return_value = mock_stats
            mock_store.__enter__ = MagicMock(return_value=mock_store)
            mock_store.__exit__ = MagicMock(return_value=False)
            MockTCA.return_value = mock_store

            result = await _fn(get_tca_report)()

        assert result["success"] is True
        assert result["avg_slippage_bps"] == 2.5

    @pytest.mark.asyncio
    async def test_tca_error(self):
        """When TCAStore raises, returns error."""
        from quantstack.mcp.tools.intraday import get_tca_report

        with patch("quantstack.mcp.tools.intraday.TCAStore", side_effect=Exception("DB error")):
            result = await _fn(get_tca_report)()

        assert result["success"] is False
        assert "error" in result


# ===========================================================================
# intraday.py — get_algo_recommendation
# ===========================================================================


class TestGetAlgoRecommendation:

    @pytest.mark.asyncio
    async def test_happy_path(self):
        """select_algo returns a recommendation."""
        from quantstack.mcp.tools.intraday import get_algo_recommendation

        @dataclass
        class MockForecast:
            spread_cost_bps: float = 1.0
            market_impact_bps: float = 2.0
            timing_cost_bps: float = 0.5
            commission_bps: float = 0.3
            total_expected_bps: float = 3.8
            participation_rate: float = 0.02
            is_liquid: bool = True
            recommended_algo: MagicMock = None
            algo_rationale: str = "test"
            min_alpha_bps: float = 5.0

        forecast = MockForecast()
        forecast.recommended_algo = MagicMock()
        forecast.recommended_algo.value = "TWAP"

        @dataclass
        class MockRec:
            recommended_algo: str = "TWAP"
            limit_price: float | None = None
            urgency: str = "normal"
            expected_slippage_bps: float = 2.0
            expected_total_cost_bps: float = 3.8
            override_reason: str | None = None
            execution_window: str = "10:00-15:30 ET"
            tca_forecast: object = None

        rec = MockRec(tca_forecast=forecast)

        with patch("quantstack.mcp.tools.intraday.select_algo", return_value=rec):
            result = await _fn(get_algo_recommendation)(
                symbol="SPY",
                side="buy",
                shares=100,
                current_price=450.0,
                adv=50_000_000,
                daily_vol_pct=1.2,
            )

        assert result["success"] is True
        assert result["recommended_algo"] == "TWAP"
        assert result["tca_forecast"] is not None

    @pytest.mark.asyncio
    async def test_algo_error(self):
        """When select_algo raises, returns error."""
        from quantstack.mcp.tools.intraday import get_algo_recommendation

        with patch("quantstack.mcp.tools.intraday.select_algo", side_effect=ValueError("bad input")):
            result = await _fn(get_algo_recommendation)(
                symbol="SPY",
                side="buy",
                shares=100,
                current_price=450.0,
                adv=50_000_000,
                daily_vol_pct=1.2,
            )

        assert result["success"] is False
        assert "error" in result


# ===========================================================================
# capitulation.py — get_capitulation_score
# ===========================================================================


class TestGetCapitulationScore:

    @pytest.mark.asyncio
    async def test_happy_path(self):
        """With 252 bars of data, computes all components."""
        from quantstack.mcp.tools.capitulation import get_capitulation_score

        df = synthetic_ohlcv("RDDT", n_days=252)
        store = MagicMock()
        store.load_ohlcv.return_value = df
        store.close = MagicMock()

        with patch("quantstack.mcp.tools.capitulation._get_reader", return_value=store):
            result = await _fn(get_capitulation_score)(symbol="RDDT")

        assert "capitulation_score" in result
        assert 0 <= result["capitulation_score"] <= 1
        assert "volume_exhaustion_score" in result
        assert "support_integrity_score" in result
        assert "wvf_score" in result
        assert "exhaustion_score" in result
        assert "consecutive_down_score" in result
        assert "recommendation" in result
        assert result["recommendation"] in ("high_conviction", "watch", "not_ready")
        assert "component_weights" in result

    @pytest.mark.asyncio
    async def test_insufficient_data(self):
        """With <60 bars, returns error."""
        from quantstack.mcp.tools.capitulation import get_capitulation_score

        df = synthetic_ohlcv("SPY", n_days=30)
        store = MagicMock()
        store.load_ohlcv.return_value = df
        store.close = MagicMock()

        with patch("quantstack.mcp.tools.capitulation._get_reader", return_value=store):
            result = await _fn(get_capitulation_score)(symbol="SPY")

        assert "error" in result
        assert result["capitulation_score"] == 0.0

    @pytest.mark.asyncio
    async def test_no_data(self):
        """When load_ohlcv returns None, returns error."""
        from quantstack.mcp.tools.capitulation import get_capitulation_score

        store = MagicMock()
        store.load_ohlcv.return_value = None
        store.close = MagicMock()

        with patch("quantstack.mcp.tools.capitulation._get_reader", return_value=store):
            result = await _fn(get_capitulation_score)(symbol="NODATA")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_store_closed_on_error(self):
        """Store.close() is called even when load_ohlcv raises."""
        from quantstack.mcp.tools.capitulation import get_capitulation_score

        store = MagicMock()
        store.load_ohlcv.side_effect = RuntimeError("DB failure")
        store.close = MagicMock()

        with patch("quantstack.mcp.tools.capitulation._get_reader", return_value=store):
            result = await _fn(get_capitulation_score)(symbol="FAIL")

        store.close.assert_called_once()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_weakest_component(self):
        """_weakest_component returns the name of the lowest-scoring component."""
        from quantstack.mcp.tools.capitulation import _weakest_component

        result = {
            "volume_exhaustion_score": 0.5,
            "support_integrity_score": 0.3,
            "wvf_score": 0.8,
            "exhaustion_score": 0.1,
            "consecutive_down_score": 0.4,
        }
        assert _weakest_component(result) == "pct_r"


# ===========================================================================
# institutional_accumulation.py — get_institutional_accumulation
# ===========================================================================


class TestGetInstitutionalAccumulation:

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        """Without ALPHA_VANTAGE_API_KEY, insider score defaults to 0.5."""
        from quantstack.mcp.tools.institutional_accumulation import get_institutional_accumulation

        mock_store = MagicMock()
        mock_store.close = MagicMock()

        # collect_options_flow is imported inline — patch at its source module
        with (
            patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": ""}, clear=False),
            patch.dict(os.environ, {"FINANCIAL_DATASETS_API_KEY": ""}, clear=False),
            patch("quantstack.mcp.tools.institutional_accumulation._get_reader", return_value=mock_store),
            patch(
                "quantstack.signal_engine.collectors.options_flow.collect_options_flow",
                side_effect=Exception("no options flow"),
            ),
        ):
            result = await _fn(get_institutional_accumulation)(symbol="TEST")

        assert "accumulation_score" in result
        assert result["insider_cluster_score"] == 0.5
        assert result["recommendation"] in ("accumulating", "neutral", "distributing")

    @pytest.mark.asyncio
    async def test_with_insider_data(self):
        """With AV key and insider transactions, computes insider score."""
        from quantstack.mcp.tools.institutional_accumulation import get_institutional_accumulation

        from datetime import timedelta
        recent_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        insider_df = pd.DataFrame({
            "transaction_date": [recent_date, recent_date],
            "acquisition_or_disposition": ["A", "A"],
            "owner_name": ["John CEO", "Jane CFO"],
            "owner_title": ["CEO", "Chief Financial Officer"],
            "shares": [1000, 500],
        })

        mock_av = MagicMock()
        mock_av.fetch_insider_transactions.return_value = insider_df

        mock_store = MagicMock()
        mock_store.close = MagicMock()

        with (
            patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test_key"}, clear=False),
            patch.dict(os.environ, {"FINANCIAL_DATASETS_API_KEY": ""}, clear=False),
            patch("quantstack.mcp.tools.institutional_accumulation.AlphaVantageClient", return_value=mock_av),
            patch("quantstack.mcp.tools.institutional_accumulation._get_reader", return_value=mock_store),
            patch(
                "quantstack.signal_engine.collectors.options_flow.collect_options_flow",
                side_effect=Exception("no options"),
            ),
        ):
            result = await _fn(get_institutional_accumulation)(symbol="NVDA")

        assert "accumulation_score" in result
        assert result["insider_distinct_buyers"] >= 1
        assert len(result["insider_details"]) > 0


# ===========================================================================
# cross_domain.py — get_cross_domain_intel
# ===========================================================================


class TestGetCrossDomainIntel:

    @pytest.mark.asyncio
    async def test_no_ctx(self):
        from quantstack.mcp.tools.cross_domain import get_cross_domain_intel

        err = {"success": False, "error": "not initialized"}
        with patch("quantstack.mcp.tools.cross_domain.live_db_or_error", return_value=(None, err)):
            result = await _fn(get_cross_domain_intel)()

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_empty_database(self):
        """With no alerts, returns empty intel."""
        from quantstack.mcp.tools.cross_domain import get_cross_domain_intel

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_conn.execute.return_value.fetchone.return_value = (0,)

        with (
            patch("quantstack.mcp.tools.cross_domain.live_db_or_error", return_value=(MagicMock(), None)),
            patch("quantstack.mcp.tools.cross_domain.pg_conn") as mock_pg,
        ):
            mock_pg.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_pg.return_value.__exit__ = MagicMock(return_value=False)

            result = await _fn(get_cross_domain_intel)()

        assert result["success"] is True
        assert result["intel_count"] == 0

    @pytest.mark.asyncio
    async def test_convergence_analysis(self):
        """_compute_convergence correctly identifies alignment."""
        from quantstack.mcp.tools.cross_domain import _compute_convergence

        items = [
            {"source_domain": "equity_investment", "intel_type": "thesis_status", "data": {"thesis_status": "intact"}},
            {"source_domain": "equity_swing", "intel_type": "momentum_signal", "data": {"direction": "bullish"}},
        ]
        conv = _compute_convergence("SPY", items)
        assert conv["alignment"] == "bullish"
        assert conv["signal_count"] == 2

    @pytest.mark.asyncio
    async def test_convergence_mixed(self):
        """Mixed signals produce 'mixed' alignment."""
        from quantstack.mcp.tools.cross_domain import _compute_convergence

        items = [
            {"source_domain": "equity_investment", "intel_type": "thesis_status", "data": {"thesis_status": "broken"}},
            {"source_domain": "equity_swing", "intel_type": "momentum_signal", "data": {"direction": "bullish"}},
        ]
        conv = _compute_convergence("SPY", items)
        assert conv["alignment"] == "mixed"
        assert len(conv["conflicts"]) > 0


# ===========================================================================
# options_execution.py — execute_options_trade
# ===========================================================================


class TestExecuteOptionsTrade:

    @pytest.mark.asyncio
    async def test_no_ctx(self):
        from quantstack.mcp.tools.options_execution import execute_options_trade

        err = {"success": False, "error": "not initialized"}
        with patch("quantstack.mcp.tools.options_execution.live_db_or_error", return_value=(None, err)):
            result = await _fn(execute_options_trade)(
                symbol="SPY", option_type="call", strike=450.0,
                expiry_date="2025-06-20", action="buy", contracts=1,
                reasoning="test", confidence=0.7,
            )

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_invalid_expiry(self):
        """Invalid expiry_date returns error."""
        from quantstack.mcp.tools.options_execution import execute_options_trade

        mock_ctx = MagicMock()
        mock_ctx.kill_switch.guard.return_value = None

        with patch("quantstack.mcp.tools.options_execution.live_db_or_error", return_value=(mock_ctx, None)):
            result = await _fn(execute_options_trade)(
                symbol="SPY", option_type="call", strike=450.0,
                expiry_date="not-a-date", action="buy", contracts=1,
                reasoning="test", confidence=0.7,
            )

        assert result["success"] is False
        assert "Invalid expiry_date" in result["error"]

    @pytest.mark.asyncio
    async def test_past_expiry(self):
        """Past expiry date returns error."""
        from quantstack.mcp.tools.options_execution import execute_options_trade

        mock_ctx = MagicMock()
        mock_ctx.kill_switch.guard.return_value = None

        with patch("quantstack.mcp.tools.options_execution.live_db_or_error", return_value=(mock_ctx, None)):
            result = await _fn(execute_options_trade)(
                symbol="SPY", option_type="call", strike=450.0,
                expiry_date="2020-01-01", action="buy", contracts=1,
                reasoning="test", confidence=0.7,
            )

        assert result["success"] is False
        assert "past" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_bs_price_calculation(self):
        """Black-Scholes helper produces valid prices."""
        from quantstack.mcp.tools.options_execution import _bs_price, _bs_delta

        # ATM call, 30 days, 25% vol
        call_price = _bs_price(100, 100, 30 / 365, 0.045, 0.25, "call")
        assert call_price > 0
        assert call_price < 100  # premium < underlying

        put_price = _bs_price(100, 100, 30 / 365, 0.045, 0.25, "put")
        assert put_price > 0

        # Deep ITM call delta ~ 1
        delta = _bs_delta(100, 50, 30 / 365, 0.045, 0.25, "call")
        assert delta > 0.95

        # Deep OTM put delta ~ 0
        delta = _bs_delta(100, 50, 30 / 365, 0.045, 0.25, "put")
        assert delta > -0.05

    @pytest.mark.asyncio
    async def test_bs_expired_option(self):
        """At expiry (T=0), BS returns intrinsic value."""
        from quantstack.mcp.tools.options_execution import _bs_price

        # ITM call at expiry
        assert _bs_price(110, 100, 0, 0.045, 0.25, "call") == 10.0
        # OTM call at expiry
        assert _bs_price(90, 100, 0, 0.045, 0.25, "call") == 0.0

    @pytest.mark.asyncio
    async def test_paper_mode_happy_path(self):
        """Paper mode: complete options trade flow."""
        from quantstack.mcp.tools.options_execution import execute_options_trade

        mock_pos = MagicMock()
        mock_pos.current_price = 450.0

        mock_snapshot = MagicMock()
        mock_snapshot.total_equity = 100_000.0

        mock_ctx = MagicMock()
        mock_ctx.kill_switch.guard.return_value = None
        mock_ctx.portfolio.get_snapshot.return_value = mock_snapshot
        mock_ctx.portfolio.get_position.return_value = mock_pos
        mock_ctx.session_id = "test-session"
        mock_ctx.conn.execute.return_value.fetchall.return_value = [
            (float(450 + i * 0.1),) for i in range(21)
        ]

        future_expiry = date.today() + __import__("datetime").timedelta(days=30)

        with (
            patch("quantstack.mcp.tools.options_execution.live_db_or_error", return_value=(mock_ctx, None)),
            patch("quantstack.mcp.tools.options_execution._estimate_vol", return_value=0.25),
            patch("quantstack.mcp.tools.options_execution.get_broker_mode", return_value="paper"),
            patch("quantstack.mcp.tools.options_execution._serialize", return_value={"cash": 100_000}),
            patch.dict(os.environ, {"RISK_MAX_PREMIUM_AT_RISK_PCT": "0.10"}, clear=False),
        ):
            result = await _fn(execute_options_trade)(
                symbol="SPY", option_type="call", strike=450.0,
                expiry_date=future_expiry.isoformat(), action="buy", contracts=1,
                reasoning="Test trade", confidence=0.7,
            )

        assert result["success"] is True
        assert result["risk_approved"] is True
        assert result["execution_mode"] == "paper"
        assert result["dte"] > 0

    @pytest.mark.asyncio
    async def test_live_mode_rejected_without_env(self):
        """Live mode without USE_REAL_TRADING returns error."""
        from quantstack.mcp.tools.options_execution import execute_options_trade

        mock_ctx = MagicMock()
        mock_ctx.kill_switch.guard.return_value = None

        future_expiry = date.today() + __import__("datetime").timedelta(days=30)

        with (
            patch("quantstack.mcp.tools.options_execution.live_db_or_error", return_value=(mock_ctx, None)),
            patch.dict(os.environ, {"USE_REAL_TRADING": "false"}, clear=False),
        ):
            result = await _fn(execute_options_trade)(
                symbol="SPY", option_type="call", strike=450.0,
                expiry_date=future_expiry.isoformat(), action="buy", contracts=1,
                reasoning="Test", confidence=0.7, paper_mode=False,
            )

        assert result["success"] is False
        assert "USE_REAL_TRADING" in result["error"]
