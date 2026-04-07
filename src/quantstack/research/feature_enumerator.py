# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Phase 1 of the autonomous feature factory: candidate enumeration.

Generates feature candidates from base features via two paths:
  1. Programmatic — deterministic transforms (lags, rolling stats, cross-interactions)
  2. LLM-assisted — Haiku generates novel feature ideas given regime context

Each candidate is a dict with keys:
  feature_id, feature_name, definition, source
"""

from __future__ import annotations

import hashlib
import itertools
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Programmatic enumeration
# ---------------------------------------------------------------------------

_LAG_PERIODS = [1, 2, 3, 5, 10, 21]
_ROLLING_WINDOWS = [5, 10, 21, 63]
_ROLLING_STATS = ["mean", "std", "skew", "zscore"]


def _feature_id(name: str) -> str:
    """Deterministic short hash for deduplication."""
    return hashlib.sha256(name.encode()).hexdigest()[:12]


def enumerate_programmatic(base_features: list[str]) -> list[dict[str, str]]:
    """Generate feature candidates from deterministic transforms.

    Transforms applied per base feature:
      - Lags: 1, 2, 3, 5, 10, 21 periods
      - Rolling stats: mean, std, skew, zscore over windows 5, 10, 21, 63
      - Cross-interactions: ratio of each unique pair

    Args:
        base_features: List of base feature names (e.g. ["close", "volume", "rsi_14"]).

    Returns:
        List of candidate dicts with feature_id, feature_name, definition, source.
    """
    candidates: list[dict[str, str]] = []

    for feat in base_features:
        # Lags
        for lag in _LAG_PERIODS:
            name = f"{feat}_lag{lag}"
            candidates.append({
                "feature_id": _feature_id(name),
                "feature_name": name,
                "definition": f"lag({feat}, {lag})",
                "source": "programmatic_lag",
            })

        # Rolling stats
        for window in _ROLLING_WINDOWS:
            for stat in _ROLLING_STATS:
                name = f"{feat}_{stat}{window}"
                candidates.append({
                    "feature_id": _feature_id(name),
                    "feature_name": name,
                    "definition": f"rolling_{stat}({feat}, {window})",
                    "source": f"programmatic_rolling_{stat}",
                })

    # Cross-interactions: ratio pairs
    for feat_a, feat_b in itertools.combinations(base_features, 2):
        name = f"{feat_a}_div_{feat_b}"
        candidates.append({
            "feature_id": _feature_id(name),
            "feature_name": name,
            "definition": f"ratio({feat_a}, {feat_b})",
            "source": "programmatic_cross",
        })

    logger.info(
        "[FeatureEnumerator] Programmatic: %d candidates from %d base features",
        len(candidates), len(base_features),
    )
    return candidates


# ---------------------------------------------------------------------------
# LLM-assisted enumeration
# ---------------------------------------------------------------------------


def enumerate_with_llm(
    base_features: list[str],
    regime: str,
) -> list[dict[str, str]]:
    """Ask Haiku for novel feature ideas given current regime.

    Falls back to an empty list on any failure — the pipeline never blocks
    on LLM availability.

    Args:
        base_features: Available base feature names.
        regime: Current market regime (e.g. "trending_up", "ranging").

    Returns:
        List of candidate dicts, or empty list on failure.
    """
    try:
        from quantstack.graphs.config import get_llm

        llm = get_llm(tier="research")
        prompt = (
            f"You are a quantitative researcher. Given these base features:\n"
            f"{base_features}\n\n"
            f"And the current market regime: {regime}\n\n"
            f"Suggest 20 novel derived features that could have predictive power "
            f"for forward returns. For each, provide:\n"
            f"- name: short snake_case name\n"
            f"- definition: mathematical definition using the base features\n\n"
            f"Return ONLY a JSON array of objects with 'name' and 'definition' keys."
        )

        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        import json

        # Try to extract JSON from the response
        # Handle cases where LLM wraps in markdown code blocks
        text = content.strip()
        if "```" in text:
            # Extract content between first ``` and last ```
            parts = text.split("```")
            for part in parts[1:]:
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                if cleaned.startswith("["):
                    text = cleaned
                    break

        parsed = json.loads(text)
        if not isinstance(parsed, list):
            return []

        candidates: list[dict[str, str]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "").strip()
            definition = item.get("definition", "").strip()
            if name and definition:
                candidates.append({
                    "feature_id": _feature_id(f"llm_{name}"),
                    "feature_name": name,
                    "definition": definition,
                    "source": "llm_haiku",
                })

        logger.info(
            "[FeatureEnumerator] LLM: %d candidates for regime=%s",
            len(candidates), regime,
        )
        return candidates

    except Exception as exc:
        logger.warning(
            "[FeatureEnumerator] LLM enumeration failed (falling back to empty): %s",
            exc,
        )
        return []
