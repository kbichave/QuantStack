# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for the Alpha Knowledge Graph.

Defines structured types for hypothesis novelty checks, factor overlap
analysis, and research history queries — the read-side of the KG.
"""

from __future__ import annotations

from pydantic import BaseModel


class SimilarHypothesis(BaseModel):
    """A hypothesis found by semantic similarity search."""

    hypothesis_id: str
    name: str
    cosine_similarity: float
    regime: str | None = None
    outcome: str | None = None


class NoveltyResult(BaseModel):
    """Result of checking whether a hypothesis is novel."""

    is_novel: bool
    similar_hypotheses: list[SimilarHypothesis]
    recommendation: str  # "novel", "redundant", "similar_but_different_regime"


class OverlapResult(BaseModel):
    """Result of checking factor overlap for a strategy."""

    is_crowded: bool
    shared_factor_count: int
    shared_factors: list[str]
    affected_strategies: list[str]


class HypothesisResult(BaseModel):
    """A single hypothesis test result from research history."""

    hypothesis_id: str
    hypothesis_text: str
    test_date: str
    outcome: str
    result_sharpe: float | None = None
    result_ic: float | None = None
    regime_at_test: str | None = None
