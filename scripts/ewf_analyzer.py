"""
EWF Chart Analyzer — extracts structured Elliott Wave data from scraped chart images.

Reads chart images from data/ewf/, calls Claude Sonnet via litellm with OHLCV
grounding, parses structured JSON responses, and writes results to
ewf_chart_analyses in PostgreSQL.

Usage:
    python scripts/ewf_analyzer.py [--date YYYY-MM-DD] [--update-type 4h] [--symbol AAPL] [--dry-run]
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm
import psycopg
from loguru import logger

from quantstack.db import pg_conn
from quantstack.llm.provider import get_model_with_fallback

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ewf"


def _resolve_model() -> str:
    """Resolve model via env override or the central LLM provider (Bedrock by default)."""
    override = os.getenv("EWF_ANALYZER_MODEL")
    if override:
        return override
    try:
        return get_model_with_fallback("heavy")
    except Exception:
        return "claude-sonnet-4-5"


MODEL: str | None = None  # resolved lazily in main()


def _get_model() -> str:
    """Return the resolved model, falling back to _resolve_model() if main() hasn't run."""
    if MODEL is not None:
        return MODEL
    return _resolve_model()
OHLCV_DAILY_TIMEFRAME = "daily"  # verified via SELECT DISTINCT timeframe FROM ohlcv

_ALL_UPDATE_TYPES = [
    "1h_premarket", "1h_midday", "4h", "daily", "weekly",
    "blue_box", "market_overview",
]

# ---------------------------------------------------------------------------
# PID collision guard
# ---------------------------------------------------------------------------


def _acquire_pid_lock(update_type: str) -> Path:
    """Write a PID file. Exit if another instance is running."""
    pid_path = Path(f"/tmp/ewf_analyzer_{update_type}.pid")
    if pid_path.exists():
        try:
            old_pid = int(pid_path.read_text().strip())
            # Check if the old process is still running
            os.kill(old_pid, 0)
            logger.warning(
                "[ewf_analyzer] PID %d still running for %s — exiting",
                old_pid, update_type,
            )
            sys.exit(0)
        except (ProcessLookupError, ValueError):
            # Process is dead or PID file is corrupt — overwrite
            pass
        except PermissionError:
            # Process exists but we can't signal it — treat as running
            logger.warning(
                "[ewf_analyzer] PID file exists for %s, cannot check — exiting",
                update_type,
            )
            sys.exit(0)
    pid_path.write_text(str(os.getpid()))
    return pid_path


def _release_pid_lock(pid_path: Path) -> None:
    """Remove the PID file if it still points to our PID."""
    try:
        if pid_path.exists():
            stored_pid = int(pid_path.read_text().strip())
            if stored_pid == os.getpid():
                pid_path.unlink()
    except Exception as exc:
        logger.debug("[ewf_analyzer] PID lock cleanup failed: %s", exc)


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------


def _get_images_to_analyze(
    date_str: str,
    update_type: str | None,
    symbol_filter: str | None,
    conn,
) -> list[dict]:
    """Return list of images that need analysis.

    Each returned dict has keys: symbol, timeframe, fetched_at, image_path, date.
    Skips images that already have a row in ewf_chart_analyses.
    """
    date_dir = DATA_DIR / date_str
    if not date_dir.is_dir():
        logger.info("[ewf_analyzer] No data directory: %s", date_dir)
        return []

    update_types = [update_type] if update_type else _ALL_UPDATE_TYPES
    images = []

    for ut in update_types:
        ut_dir = date_dir / ut
        if not ut_dir.is_dir():
            continue

        # Read metadata.json for fetched_at timestamp
        metadata_path = ut_dir / "metadata.json"
        fetched_at = None
        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text())
                fetched_at_str = meta.get("fetched_at_utc", "")
                if fetched_at_str:
                    fetched_at = datetime.fromisoformat(fetched_at_str).replace(
                        tzinfo=timezone.utc
                    )
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "[ewf_analyzer] Failed to parse metadata.json in %s: %s",
                    ut_dir, exc,
                )

        # Discover image files
        for img_file in sorted(ut_dir.iterdir()):
            if img_file.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue

            sym = img_file.stem.upper()
            if symbol_filter and sym != symbol_filter.upper():
                continue

            # For market_overview images, use sentinel symbol
            if ut == "market_overview":
                sym = "$MKT"

            # Use file mtime as fallback fetched_at
            img_fetched_at = fetched_at or datetime.fromtimestamp(
                img_file.stat().st_mtime, tz=timezone.utc
            )

            # Check if analysis already exists in DB
            try:
                conn.execute(
                    "SELECT 1 FROM ewf_chart_analyses "
                    "WHERE symbol = %s AND timeframe = %s AND fetched_at = %s",
                    (sym, ut, img_fetched_at),
                )
                if conn.fetchone() is not None:
                    continue  # Already analyzed
            except Exception as exc:
                logger.warning(
                    "[ewf_analyzer] DB check failed for %s/%s: %s", sym, ut, exc
                )

            images.append({
                "symbol": sym,
                "timeframe": ut,
                "fetched_at": img_fetched_at,
                "image_path": str(img_file),
                "date": date_str,
            })

    return images


# ---------------------------------------------------------------------------
# OHLCV grounding query
# ---------------------------------------------------------------------------


def _query_ohlcv_context(symbol: str, conn) -> list[tuple[str, float]]:
    """Return up to 30 (date, close) tuples for symbol, most recent last."""
    if symbol == "$MKT":
        return []  # No OHLCV for market overview
    try:
        conn.execute(
            "SELECT timestamp::date, close FROM ohlcv "
            "WHERE symbol = %s AND timeframe = %s "
            "ORDER BY timestamp DESC LIMIT 30",
            (symbol, OHLCV_DAILY_TIMEFRAME),
        )
        rows = conn.fetchall()
        # Reverse to most-recent-last for prompt readability
        return [(str(r[0]), r[1]) for r in reversed(rows)]
    except Exception as exc:
        logger.warning("[ewf_analyzer] OHLCV query failed for %s: %s", symbol, exc)
        return []


# ---------------------------------------------------------------------------
# Vision prompts and structured output schema
# ---------------------------------------------------------------------------

EWF_SYSTEM_PROMPT = """\
You are an expert at reading Elliott Wave Forecast (EWF) chart images and extracting \
structured data from the analyst's annotations. You are NOT deriving your own wave \
count — you are reading what the EWF analyst has already drawn on the chart.

## EWF Visual Vocabulary

EWF charts use a consistent annotation system. Learn to recognize each element:

### Wave Labels (text on or near price bars)
- **Red labels** = higher-degree waves (Primary or Cycle). Examples: red 1, 2, 3, 4, 5 \
or red A, B, C or red W, X, Y.
- **Blue labels** = lower-degree waves (Intermediate, Minor, Minute). Examples: blue \
((i)), ((ii)), ((iii)), ((iv)), ((v)) or blue ((a)), ((b)), ((c)).
- **Degree bracket notation** (highest to lowest):
  - Roman numerals: I, II, III, IV, V (Cycle degree)
  - Plain numbers/letters: 1, 2, 3, 4, 5 or A, B, C (Primary degree)
  - Single parentheses: (1), (2), (A), (B), (W), (X), (Y) (Intermediate degree)
  - Double parentheses: ((1)), ((2)), ((a)), ((b)), ((i)), ((v)) (Minor/Minute degree)
- **Complex corrections** use W-X-Y or W-X-Y-X-Z labeling for double/triple zigzags.

### Directional Signal — THE MOST IMPORTANT ELEMENT
- A **boxed label** usually in the bottom-right or top-right corner of the chart.
- **"Turning Up" with an upward arrow (↗)** = bullish — analyst expects price to rally.
- **"Turning Down" with a downward arrow (↘)** = bearish — analyst expects price to decline.
- If neither box is present, the signal is "none".
- This is the single most actionable annotation on the chart. Always look for it FIRST.

### Invalidation Level (horizontal line + text label)
- A **green or red horizontal line** labeled "Invalidation level" or "Invalidation Level".
- The price is printed near the line on the y-axis. Read the EXACT price from the y-axis scale.
- If price crosses this level, the analyst's wave count is invalid.

### Projected Path (dashed/dotted lines on the RIGHT side of the chart)
- **Dashed diagonal lines** extending to the right of the current price = the analyst's \
projected future path.
- These lines show expected future wave movements with labels at projected turning points.
- CRITICAL: Distinguish projected labels (on dashed lines) from completed labels (on solid \
candlestick bars). Only completed labels describe what has already happened.

### Standard Boilerplate (IGNORE these — they are NOT signals)
- **"We Do Not Recommend Selling"** (red text) = standard EWF disclaimer on every chart.
- **Disclaimer box** about Blue Boxes and right-side tags = boilerplate, present on every chart.

### Blue Box (rare — only on some charts)
- A **shaded blue/gray rectangle** marking a high-probability reversal zone (100%-161.8% \
Fibonacci extension of a corrective sequence).
- When price enters the Blue Box, a reversal is expected.

## How to Read the Chart (follow this procedure step by step)

1. **Read the y-axis scale**: Note the min/max price, gridline spacing, and the highlighted \
current price on the right edge of the chart.
2. **Find the Turning Signal**: Scan corners for the boxed "Turning Up ↗" or "Turning Down ↘". \
This determines the bias field.
3. **Find the Invalidation Level**: Locate the horizontal line labeled "Invalidation". Read its \
EXACT price from the y-axis. Cross-reference against the OHLCV table if provided.
4. **Read current price**: The rightmost candlestick's close is typically highlighted on the \
y-axis right edge (often in a colored box showing the live price).
5. **Read completed wave labels**: Working LEFT from the current bar, identify the last 3-5 wave \
labels placed ON actual price bars. Note each label, its degree notation, and its approximate \
price level by reading across to the y-axis.
6. **Read the projected path**: Look at dashed lines extending RIGHT of the last bar. Note the \
projected wave labels and their approximate target prices from the y-axis.
7. **Cross-reference with OHLCV**: If an OHLCV price table is provided, verify your y-axis \
readings are in the correct range. If your readings differ by more than 2-3%, re-read the y-axis.
8. **Check for Blue Box**: If a shaded rectangular zone exists, note its upper and lower bounds.

## Elliott Wave Rules (for validation only)
- Impulse: 5 waves (1-5). Wave 3 is typically longest. Wave 2 never retraces 100%+ of Wave 1. \
Wave 4 never overlaps Wave 1 price territory.
- Corrective: 3 waves (A,B,C) or complex W-X-Y. Common after a completed impulse.
- If the analyst's count on the chart violates these rules, set invalidation_rule_violated=true.

## Output
Use the `reasoning` field to show your step-by-step chart reading BEFORE filling structured \
fields. Describe what you see for each step above. Then fill all other fields based on your \
reasoning."""

EWF_ANALYSIS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_ewf_analysis",
        "description": "Submit the structured Elliott Wave analysis extracted from the chart.",
        "parameters": {
            "type": "object",
            "required": [
                "reasoning", "symbol", "timeframe", "bias", "turning_signal",
                "wave_position", "wave_degree", "current_wave_label",
                "completed_wave_sequence", "projected_path",
                "key_levels", "blue_box_active", "blue_box_zone",
                "confidence", "invalidation_rule_violated", "analyst_notes", "summary",
            ],
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Step-by-step chain of thought describing what you see on the chart. "
                        "Must cover: (1) y-axis scale and current price, (2) Turning Signal "
                        "location and text, (3) invalidation level price, (4) completed wave "
                        "labels with prices, (5) projected path labels with targets, "
                        "(6) OHLCV cross-reference check. Be specific about price readings."
                    ),
                },
                "symbol": {"type": "string"},
                "timeframe": {"type": "string"},
                "bias": {
                    "type": "string",
                    "enum": ["bullish", "bearish", "neutral", "unknown"],
                    "description": "Directional bias — must match the Turning Signal on the chart.",
                },
                "turning_signal": {
                    "type": "string",
                    "enum": ["turning_up", "turning_down", "none"],
                    "description": (
                        "The boxed Turning Up/Down signal on the chart. "
                        "'turning_up' = bullish, 'turning_down' = bearish, "
                        "'none' = no turning signal visible."
                    ),
                },
                "wave_position": {
                    "type": "string",
                    "description": (
                        "Current wave position in plain English, e.g. "
                        "'completing wave C of (Y) of ((2))' or "
                        "'in wave (3) up after wave (2) correction'."
                    ),
                },
                "wave_degree": {
                    "type": "string",
                    "enum": [
                        "subminuette", "minuette", "minute", "minor",
                        "intermediate", "primary", "cycle", "unknown",
                    ],
                    "description": "The highest degree wave currently in progress.",
                },
                "current_wave_label": {
                    "type": "string",
                    "description": (
                        "The most recent completed wave label on an actual price bar "
                        "(NOT on a projected dashed line). E.g. 'C', '(2)', '((v))'."
                    ),
                },
                "completed_wave_sequence": {
                    "type": "string",
                    "description": (
                        "The last 3-5 completed wave labels in order, separated by ' → '. "
                        "Include degree notation. E.g. '((iii)) → ((iv)) → ((v)) → (W) → C'. "
                        "Only labels on actual price bars, NOT projected labels."
                    ),
                },
                "projected_path": {
                    "type": "string",
                    "description": (
                        "The analyst's projected future path shown by dashed lines. "
                        "Describe the expected sequence and approximate target prices. "
                        "E.g. 'wave (2) correction to ~172, then wave (3) up toward ~192'. "
                        "Use 'no projection visible' if no dashed lines present."
                    ),
                },
                "key_levels": {
                    "type": "object",
                    "properties": {
                        "support": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Support levels read from the y-axis.",
                        },
                        "resistance": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Resistance levels read from the y-axis.",
                        },
                        "invalidation": {
                            "type": ["number", "null"],
                            "description": (
                                "The exact invalidation level price from the labeled "
                                "horizontal line. Read directly from the y-axis."
                            ),
                        },
                        "target": {
                            "type": ["number", "null"],
                            "description": (
                                "The primary projected target price from the highest "
                                "point of the projected path."
                            ),
                        },
                    },
                    "required": ["support", "resistance", "invalidation", "target"],
                },
                "blue_box_active": {"type": "boolean"},
                "blue_box_zone": {
                    "oneOf": [
                        {"type": "null"},
                        {
                            "type": "object",
                            "properties": {
                                "low": {"type": "number"},
                                "high": {"type": "number"},
                            },
                            "required": ["low", "high"],
                        },
                    ],
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": (
                        "Your confidence in this extraction. Lower if: chart is cluttered, "
                        "labels are hard to read, y-axis prices are ambiguous, or OHLCV "
                        "cross-reference shows discrepancies."
                    ),
                },
                "invalidation_rule_violated": {"type": "boolean"},
                "analyst_notes": {
                    "type": "string",
                    "description": (
                        "Any additional observations: alternative counts shown, "
                        "unusual annotations, 'Fake Wick' labels, etc."
                    ),
                },
                "summary": {
                    "type": "string",
                    "maxLength": 300,
                    "description": (
                        "One-paragraph summary: turning signal, current position, "
                        "invalidation level, and projected target. Be specific about prices."
                    ),
                },
            },
        },
    },
}


def _build_vision_prompt(
    symbol: str,
    timeframe: str,
    ohlcv_rows: list[tuple[str, float]],
) -> tuple[str, str]:
    """Build system + user prompt for the vision model.

    Returns (system_prompt, user_text_prompt).
    """
    price_section = ""
    if ohlcv_rows:
        lines = ["Date        | Close"]
        lines.append("------------|-------")
        for dt, close in ohlcv_rows:
            lines.append(f"{dt}  | {close:.2f}")
        last_close = ohlcv_rows[-1][1]
        price_section = (
            f"\n\n## OHLCV Price Reference for {symbol}\n"
            f"Last {len(ohlcv_rows)} daily closes (most recent last):\n"
            + "\n".join(lines)
            + f"\n\nMost recent close: {last_close:.2f} — use this to calibrate "
            f"your y-axis readings. If the chart's highlighted price differs from "
            f"this by more than 2-3%, re-check your y-axis reading."
        )
    else:
        price_section = (
            "\n\nNo OHLCV data available for price calibration. "
            "Read prices carefully from the y-axis scale and gridlines."
        )

    user_prompt = (
        f"Read this EWF chart for **{symbol}** ({timeframe} timeframe).\n\n"
        f"Follow the reading procedure from your instructions:\n"
        f"1. Read the y-axis scale and current price\n"
        f"2. Find the Turning Up/Down signal box\n"
        f"3. Find the invalidation level line and price\n"
        f"4. Read completed wave labels on actual price bars\n"
        f"5. Read projected path (dashed lines) with target prices\n"
        f"6. Cross-reference prices against OHLCV data below\n\n"
        f"Put your step-by-step observations in the `reasoning` field, "
        f"then fill all structured fields."
        f"{price_section}"
    )
    return EWF_SYSTEM_PROMPT, user_prompt


def _detect_mime_type(image_path: Path) -> str:
    """Detect MIME type for image, fallback to image/jpeg."""
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime and mime.startswith("image/"):
        return mime
    return "image/jpeg"


def _analyze_image(
    image_path: Path,
    symbol: str,
    timeframe: str,
    ohlcv_rows: list[tuple[str, float]],
) -> dict:
    """Call Claude Sonnet via litellm with the chart image and OHLCV context.

    Returns a dict matching ewf_chart_analyses columns.
    On any exception, returns a minimal fallback dict.
    """
    try:
        image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        mime_type = _detect_mime_type(image_path)

        system_prompt, user_prompt = _build_vision_prompt(
            symbol, timeframe, ohlcv_rows
        )

        model = _get_model()
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}",
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            tools=[EWF_ANALYSIS_SCHEMA],
            tool_choice={"type": "function", "function": {"name": "submit_ewf_analysis"}},
            max_tokens=4000,
            temperature=0.0,
        )

        # Extract from tool call if present, else fall back to content
        msg = response.choices[0].message
        if msg.tool_calls:
            raw_text = msg.tool_calls[0].function.arguments
        else:
            raw_text = msg.content or ""
        result = _parse_vision_response(raw_text, symbol, timeframe)
        result["raw_analysis"] = raw_text
        result["model_used"] = model
        result["image_path"] = str(image_path.relative_to(DATA_DIR.parent.parent))
        if not ohlcv_rows:
            notes = result.get("analyst_notes") or ""
            if "OHLCV" not in notes and "ohlcv" not in notes:
                result["analyst_notes"] = (
                    f"No OHLCV data for price calibration. {notes}"
                ).strip()
        return result

    except Exception as exc:
        logger.warning(
            "[ewf_analyzer] API call failed for %s/%s (%s): %s",
            symbol, timeframe, image_path.name, exc,
        )
        return _fallback_result(symbol, timeframe, str(exc))


def _parse_vision_response(raw: str, symbol: str, timeframe: str) -> dict:
    """Parse the JSON string returned by the vision model. Never raises."""
    try:
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        data = json.loads(text)
        return {
            "bias": data.get("bias", "unknown"),
            "turning_signal": data.get("turning_signal", "none"),
            "wave_position": data.get("wave_position"),
            "wave_degree": data.get("wave_degree", "unknown"),
            "current_wave_label": data.get("current_wave_label"),
            "completed_wave_sequence": data.get("completed_wave_sequence"),
            "projected_path": data.get("projected_path"),
            "key_levels": data.get("key_levels"),
            "blue_box_active": bool(data.get("blue_box_active", False)),
            "blue_box_zone": data.get("blue_box_zone"),
            "confidence": float(data.get("confidence", 0.5)),
            "invalidation_rule_violated": bool(
                data.get("invalidation_rule_violated", False)
            ),
            "analyst_notes": data.get("analyst_notes"),
            "summary": data.get("summary"),
            "reasoning": data.get("reasoning"),
        }
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(
            "[ewf_analyzer] JSON parse failed for %s/%s: %s | raw: %.200s",
            symbol, timeframe, exc, raw,
        )
        return _fallback_result(symbol, timeframe, raw)


def _fallback_result(symbol: str, timeframe: str, raw: str) -> dict:
    """Minimal fallback dict for a failed analysis."""
    return {
        "bias": "unknown",
        "turning_signal": "none",
        "wave_position": None,
        "wave_degree": "unknown",
        "current_wave_label": None,
        "completed_wave_sequence": None,
        "projected_path": None,
        "key_levels": None,
        "blue_box_active": False,
        "blue_box_zone": None,
        "confidence": 0.0,
        "invalidation_rule_violated": False,
        "analyst_notes": None,
        "summary": None,
        "reasoning": None,
        "raw_analysis": raw,
        "model_used": _get_model(),
    }


# ---------------------------------------------------------------------------
# DB upsert
# ---------------------------------------------------------------------------


def _upsert_analysis(conn, symbol: str, timeframe: str, fetched_at, row: dict) -> None:
    """Write one analysis row to ewf_chart_analyses via ON CONFLICT DO UPDATE."""
    try:
        kl = json.dumps(row.get("key_levels")) if row.get("key_levels") else None
        bbz = json.dumps(row.get("blue_box_zone")) if row.get("blue_box_zone") else None

        conn.execute(
            """
            INSERT INTO ewf_chart_analyses (
                symbol, timeframe, fetched_at, image_path,
                bias, turning_signal, wave_position, wave_degree,
                current_wave_label, completed_wave_sequence, projected_path,
                key_levels, blue_box_active, blue_box_zone,
                confidence, invalidation_rule_violated,
                analyst_notes, summary, reasoning,
                raw_analysis, model_used
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s::jsonb, %s, %s::jsonb,
                %s, %s,
                %s, %s, %s,
                %s, %s
            )
            ON CONFLICT (symbol, timeframe, fetched_at)
            DO UPDATE SET
                analyzed_at = NOW(),
                image_path = EXCLUDED.image_path,
                bias = EXCLUDED.bias,
                turning_signal = EXCLUDED.turning_signal,
                wave_position = EXCLUDED.wave_position,
                wave_degree = EXCLUDED.wave_degree,
                current_wave_label = EXCLUDED.current_wave_label,
                completed_wave_sequence = EXCLUDED.completed_wave_sequence,
                projected_path = EXCLUDED.projected_path,
                key_levels = EXCLUDED.key_levels,
                blue_box_active = EXCLUDED.blue_box_active,
                blue_box_zone = EXCLUDED.blue_box_zone,
                confidence = EXCLUDED.confidence,
                invalidation_rule_violated = EXCLUDED.invalidation_rule_violated,
                analyst_notes = EXCLUDED.analyst_notes,
                summary = EXCLUDED.summary,
                reasoning = EXCLUDED.reasoning,
                raw_analysis = EXCLUDED.raw_analysis,
                model_used = EXCLUDED.model_used
            """,
            (
                symbol, timeframe, fetched_at, row.get("image_path"),
                row.get("bias"), row.get("turning_signal", "none"),
                row.get("wave_position"), row.get("wave_degree"),
                row.get("current_wave_label"), row.get("completed_wave_sequence"),
                row.get("projected_path"),
                kl, row.get("blue_box_active", False), bbz,
                row.get("confidence"), row.get("invalidation_rule_violated", False),
                row.get("analyst_notes"), row.get("summary"),
                row.get("reasoning"),
                row.get("raw_analysis"), row.get("model_used"),
            ),
        )
        conn.commit()
    except Exception as exc:
        logger.error(
            "[ewf_analyzer] DB upsert failed for %s/%s: %s",
            symbol, timeframe, exc,
        )
        try:
            conn.rollback()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Blue Box alert handler (stub — section-04 fills this in)
# ---------------------------------------------------------------------------


_SESSION_HANDOFFS_PATH = Path(__file__).resolve().parent.parent / ".claude" / "memory" / "session_handoffs.md"


def _handle_blue_box_alerts(conn, update_type: str) -> None:
    """Query for new Blue Box analyses from the last 30 minutes and fire alerts.

    For each symbol with blue_box_active=True in a recently-analyzed row:
    1. Add symbol to screener_results as an EWF Blue Box candidate (tier 1)
    2. Upsert ewf_blue_box_alert in signal_state
    3. Append formatted entry to .claude/memory/session_handoffs.md

    Silent no-op when no qualifying rows exist. Never raises — failures are
    logged but do not propagate, since the primary DB write has already succeeded.
    """
    try:
        conn.execute(
            "SELECT symbol, bias, blue_box_zone, confidence, summary "
            "FROM ewf_chart_analyses "
            "WHERE timeframe = 'blue_box' "
            "  AND blue_box_active = TRUE "
            "  AND analyzed_at > NOW() - INTERVAL '30 minutes' "
            "ORDER BY confidence DESC"
        )
        rows = conn.fetchall()
    except Exception as exc:
        logger.warning("[ewf_analyzer] Blue Box query failed: %s", exc)
        return

    if not rows:
        return

    for symbol, bias, blue_box_zone_raw, confidence, summary in rows:
        zone = _parse_jsonb(blue_box_zone_raw)
        zone_low = zone.get("low") if zone else None
        zone_high = zone.get("high") if zone else None

        # Action 1: Watchlist via screener_results (tier 1, high score)
        try:
            conn.execute(
                "INSERT INTO screener_results "
                "(symbol, screened_at, regime_used, tier, composite_score) "
                "VALUES (%s, NOW(), 'ewf_blue_box', 1, %s)",
                (symbol, confidence or 0.5),
            )
            conn.commit()
        except Exception as exc:
            logger.warning("[ewf_analyzer] Watchlist write failed for %s: %s", symbol, exc)

        # Action 2: Signal state upsert (DELETE + INSERT since table may lack PK constraint)
        try:
            conn.execute("DELETE FROM signal_state WHERE symbol = %s", (symbol,))
            conn.execute(
                "INSERT INTO signal_state "
                "(symbol, action, confidence, position_size_pct, "
                " stop_loss, take_profit, generated_at, expires_at, session_id) "
                "VALUES (%s, 'ewf_blue_box_alert', %s, 0, %s, %s, NOW(), "
                "        NOW() + INTERVAL '24 hours', 'ewf_analyzer')",
                (symbol, confidence or 0.0, zone_low, zone_high),
            )
            conn.commit()
        except Exception as exc:
            logger.warning("[ewf_analyzer] Signal state upsert failed for %s: %s", symbol, exc)

        # Action 3: Session handoffs log
        try:
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if zone_low is not None and zone_high is not None:
                zone_str = f"{zone_low}–{zone_high}"
            else:
                zone_str = "zone unknown"
                logger.warning(
                    "[ewf_analyzer] Blue Box for %s has no zone data", symbol
                )

            conf_pct = f"{(confidence or 0):.0%}"
            entry = (
                f"\n## EWF Blue Box — {symbol} {bias} [{now_str}]\n"
                f"Zone: {zone_str} | Confidence: {conf_pct}\n"
                f"{summary or 'No summary available'}\n"
            )

            _SESSION_HANDOFFS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_SESSION_HANDOFFS_PATH, "a") as f:
                f.write(entry)
        except Exception as exc:
            logger.warning(
                "[ewf_analyzer] Session handoffs write failed for %s: %s",
                symbol, exc,
            )


def _parse_jsonb(val: Any) -> dict | None:
    """Parse a JSONB value that may be str, dict, or None."""
    if val is None:
        return None
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    global MODEL
    args = _parse_args()
    MODEL = _resolve_model()
    logger.info("[ewf_analyzer] Using model: %s", MODEL)
    date_str = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    update_type = args.update_type
    symbol_filter = args.symbol
    dry_run = args.dry_run

    lock_name = update_type or "all"
    pid_path = _acquire_pid_lock(lock_name)

    try:
        with pg_conn() as conn:
            images = _get_images_to_analyze(date_str, update_type, symbol_filter, conn)

            if not images:
                logger.info("[ewf_analyzer] No images to analyze for %s/%s", date_str, lock_name)
                return

            logger.info(
                "[ewf_analyzer] Found %d images to analyze for %s/%s",
                len(images), date_str, lock_name,
            )

            if dry_run:
                for img in images:
                    logger.info(
                        "[ewf_analyzer] [DRY-RUN] Would analyze %s/%s: %s",
                        img["symbol"], img["timeframe"], img["image_path"],
                    )
                return

            processed_update_types: set[str] = set()

            for img in images:
                try:
                    image_path = Path(img["image_path"])
                    if not image_path.exists():
                        logger.warning(
                            "[ewf_analyzer] Image not found: %s", image_path
                        )
                        continue

                    ohlcv_rows = _query_ohlcv_context(img["symbol"], conn)

                    result = _analyze_image(
                        image_path, img["symbol"], img["timeframe"], ohlcv_rows
                    )

                    _upsert_analysis(
                        conn, img["symbol"], img["timeframe"],
                        img["fetched_at"], result,
                    )

                    logger.info(
                        "[ewf_analyzer] %s/%s: bias=%s confidence=%.2f",
                        img["symbol"], img["timeframe"],
                        result.get("bias", "?"), result.get("confidence", 0),
                    )
                    processed_update_types.add(img["timeframe"])

                except Exception as exc:
                    logger.warning(
                        "[ewf_analyzer] Failed to process %s/%s: %s",
                        img["symbol"], img["timeframe"], exc,
                    )

            # Blue Box alerts after each update_type batch
            for ut in processed_update_types:
                _handle_blue_box_alerts(conn, ut)

    finally:
        _release_pid_lock(pid_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EWF Chart Analyzer")
    parser.add_argument("--date", help="Date to analyze (YYYY-MM-DD, default: today)")
    parser.add_argument("--update-type", help="Specific update type (e.g., 4h, daily)")
    parser.add_argument("--symbol", help="Specific symbol to analyze")
    parser.add_argument("--dry-run", action="store_true", help="Log only, no API calls or DB writes")
    return parser.parse_args()


if __name__ == "__main__":
    main()
