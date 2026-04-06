"""End-to-end integration and regression tests for the EWF pipeline (Section 10)."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Fixture image path
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "ewf"
FIXTURE_IMAGE = FIXTURE_DIR / "AAPL_4h.png"
FIXTURE_METADATA = FIXTURE_DIR / "metadata.json"


def _mock_litellm_response(analysis_dict: dict) -> MagicMock:
    """Build a litellm completion response mock with tool_calls."""
    tool_call = MagicMock()
    tool_call.function.arguments = json.dumps(analysis_dict)
    msg = MagicMock()
    msg.tool_calls = [tool_call]
    msg.content = None
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


_VALID_ANALYSIS = {
    "symbol": "AAPL",
    "timeframe": "4h",
    "bias": "bullish",
    "wave_position": "completing wave 3 of 5",
    "wave_degree": "minor",
    "current_wave_label": "3",
    "key_levels": {
        "support": [170.0, 165.0],
        "resistance": [185.0, 190.0],
        "invalidation": 162.0,
        "target": 188.0,
    },
    "blue_box_active": False,
    "blue_box_zone": None,
    "confidence": 0.82,
    "invalidation_rule_violated": False,
    "analyst_notes": "Clean impulse structure visible.",
    "summary": "AAPL bullish wave 3 targeting 188 with invalidation at 162.",
}


@pytest.fixture(autouse=True)
def _clean_ewf_rows(trading_ctx):
    """Clean EWF test rows before each test."""
    trading_ctx.db.execute("DELETE FROM ewf_chart_analyses WHERE symbol = 'AAPL'")
    trading_ctx.db.commit()
    yield


def _setup_ewf_data_dir(tmp_path: Path) -> Path:
    """Set up a data/ewf/{date}/4h/ directory with the fixture image and metadata."""
    date_str = "2026-04-04"
    ut_dir = tmp_path / date_str / "4h"
    ut_dir.mkdir(parents=True)
    shutil.copy2(FIXTURE_IMAGE, ut_dir / "AAPL.png")
    # Write metadata in the format the analyzer expects
    meta = {
        "update_type": "4h",
        "date": date_str,
        "fetched_at_utc": "2026-04-04T14:00:00",
        "results": {"AAPL": "ok"},
    }
    (ut_dir / "metadata.json").write_text(json.dumps(meta))
    return tmp_path


@pytest.mark.integration
def test_full_pipeline_single_image(trading_ctx, tmp_path):
    """Fixture image + mocked litellm → row in ewf_chart_analyses with correct fields."""
    from scripts.ewf_analyzer import (
        _analyze_image,
        _get_images_to_analyze,
        _upsert_analysis,
    )

    data_dir = _setup_ewf_data_dir(tmp_path)
    mock_resp = _mock_litellm_response(_VALID_ANALYSIS)

    with patch("scripts.ewf_analyzer.DATA_DIR", data_dir):
        images = _get_images_to_analyze("2026-04-04", "4h", None, trading_ctx.db)
    assert len(images) == 1

    img = images[0]
    # DATA_DIR needs to be patched so image_path.relative_to(DATA_DIR.parent.parent) works
    # DATA_DIR = tmp_path → DATA_DIR.parent.parent is tmp_path's grandparent which contains tmp_path
    with patch("scripts.ewf_analyzer.litellm.completion", return_value=mock_resp), \
         patch("scripts.ewf_analyzer.DATA_DIR", data_dir):
        result = _analyze_image(
            Path(img["image_path"]), img["symbol"], img["timeframe"], []
        )

    _upsert_analysis(trading_ctx.db, img["symbol"], img["timeframe"], img["fetched_at"], result)

    trading_ctx.db.execute(
        "SELECT bias, confidence, blue_box_active, image_path, model_used "
        "FROM ewf_chart_analyses WHERE symbol = 'AAPL' AND timeframe = '4h'"
    )
    row = trading_ctx.db.fetchone()
    assert row is not None
    bias, confidence, bb_active, image_path, model_used = row
    assert bias == "bullish"
    assert confidence == 0.82
    assert bb_active is False
    assert image_path is not None
    assert model_used is not None


@pytest.mark.integration
async def test_collector_reads_from_real_db(trading_ctx):
    """Insert fixture row directly → collect_ewf returns matching dict."""
    from quantstack.signal_engine.collectors.ewf_collector import collect_ewf

    trading_ctx.db.execute(
        """
        INSERT INTO ewf_chart_analyses
            (symbol, timeframe, fetched_at, analyzed_at, bias, wave_position,
             wave_degree, current_wave_label, key_levels, blue_box_active,
             blue_box_zone, confidence, summary)
        VALUES ('AAPL', '4h', NOW(), NOW(), 'bullish', 'wave 3 of 5',
                'minor', '3',
                '{"support": [170.0], "resistance": [185.0], "invalidation": 162.0, "target": 188.0}',
                FALSE, NULL, 0.85, 'Test summary')
        """
    )
    trading_ctx.db.commit()

    result = await collect_ewf("AAPL", store=None)
    assert isinstance(result, dict)
    assert result != {}
    assert result["ewf_bias"] == "bullish"
    assert result["ewf_confidence"] == 0.85
    assert result["ewf_timeframe_used"] == "4h"
    assert isinstance(result["ewf_age_hours"], float)
    assert result["ewf_age_hours"] < 0.1  # very fresh

    expected_keys = [
        "ewf_bias", "ewf_wave_position", "ewf_wave_degree",
        "ewf_current_wave_label", "ewf_confidence", "ewf_key_support",
        "ewf_key_resistance", "ewf_invalidation_level", "ewf_target",
        "ewf_blue_box_active", "ewf_blue_box_low", "ewf_blue_box_high",
        "ewf_summary", "ewf_timeframe_used", "ewf_age_hours",
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"


@pytest.mark.integration
def test_empty_data_dir_exits_cleanly(trading_ctx, tmp_path):
    """Empty data/ewf/ dir → analyzer processes nothing, no DB writes, no API calls."""
    from scripts.ewf_analyzer import _get_images_to_analyze

    with patch("scripts.ewf_analyzer.DATA_DIR", tmp_path / "nonexistent"):
        images = _get_images_to_analyze("2026-04-04", "4h", None, trading_ctx.db)
    assert images == []

    trading_ctx.db.execute("SELECT COUNT(*) FROM ewf_chart_analyses WHERE symbol = 'AAPL'")
    count = trading_ctx.db.fetchone()[0]
    assert count == 0


@pytest.mark.regression
def test_upsert_idempotency(trading_ctx, tmp_path):
    """Re-running analyzer on same image produces ON CONFLICT DO UPDATE: 1 row, updated analyzed_at."""
    from scripts.ewf_analyzer import _analyze_image, _get_images_to_analyze, _upsert_analysis

    data_dir = _setup_ewf_data_dir(tmp_path)
    mock_resp = _mock_litellm_response(_VALID_ANALYSIS)

    with patch("scripts.ewf_analyzer.DATA_DIR", data_dir):
        images = _get_images_to_analyze("2026-04-04", "4h", None, trading_ctx.db)
    assert len(images) == 1
    img = images[0]

    # First run
    with patch("scripts.ewf_analyzer.litellm.completion", return_value=mock_resp):
        result1 = _analyze_image(Path(img["image_path"]), img["symbol"], img["timeframe"], [])
    _upsert_analysis(trading_ctx.db, img["symbol"], img["timeframe"], img["fetched_at"], result1)

    trading_ctx.db.execute(
        "SELECT analyzed_at FROM ewf_chart_analyses "
        "WHERE symbol = 'AAPL' AND timeframe = '4h' AND fetched_at = %s",
        (img["fetched_at"],),
    )
    t1 = trading_ctx.db.fetchone()[0]

    # Second run (same image, same fetched_at)
    with patch("scripts.ewf_analyzer.litellm.completion", return_value=mock_resp):
        result2 = _analyze_image(Path(img["image_path"]), img["symbol"], img["timeframe"], [])
    _upsert_analysis(trading_ctx.db, img["symbol"], img["timeframe"], img["fetched_at"], result2)

    trading_ctx.db.execute(
        "SELECT COUNT(*) FROM ewf_chart_analyses "
        "WHERE symbol = 'AAPL' AND timeframe = '4h' AND fetched_at = %s",
        (img["fetched_at"],),
    )
    count = trading_ctx.db.fetchone()[0]
    assert count == 1

    trading_ctx.db.execute(
        "SELECT analyzed_at FROM ewf_chart_analyses "
        "WHERE symbol = 'AAPL' AND timeframe = '4h' AND fetched_at = %s",
        (img["fetched_at"],),
    )
    t2 = trading_ctx.db.fetchone()[0]
    assert t2 >= t1  # updated, not re-inserted


@pytest.mark.regression
async def test_ttl_expiry_returns_empty(trading_ctx):
    """Stale row (all TTLs expired) → collect_ewf returns {} — documents Monday morning gap.

    On Monday morning before the first EWF scraper run completes (~09:15), there may be
    no fresh EWF data. The collector returns {} (neutral). The trading graph must not
    block on EWF being present — it is one signal among many.
    """
    from quantstack.signal_engine.collectors.ewf_collector import collect_ewf

    trading_ctx.db.execute(
        """
        INSERT INTO ewf_chart_analyses
            (symbol, timeframe, fetched_at, analyzed_at, bias, confidence, summary)
        VALUES ('AAPL', '4h', NOW() - INTERVAL '9 days',
                NOW() - INTERVAL '9 days', 'bullish', 0.8, 'Old analysis')
        """
    )
    trading_ctx.db.commit()

    result = await collect_ewf("AAPL", store=None)
    assert result == {}
