# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""LLM-facing tools for the Alpha Knowledge Graph.

Four tools that wrap ``KnowledgeGraph`` methods for use by LangGraph agents.
All return JSON-serialised results for downstream LLM consumption.
"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field


@tool
def check_hypothesis_novelty(
    hypothesis_text: Annotated[str, Field(description="The hypothesis to check for novelty against existing research")],
) -> str:
    """Check whether a hypothesis is novel or has been tested before.

    Returns a JSON object with is_novel (bool), similar_hypotheses (list),
    and recommendation ('novel', 'redundant', or 'similar_but_different_regime').
    Use this BEFORE running expensive experiments to avoid redundant work.
    """
    from quantstack.knowledge.graph import KnowledgeGraph

    kg = KnowledgeGraph()
    result = kg.check_hypothesis_novelty(hypothesis_text)
    return result.model_dump_json(indent=2)


@tool
def check_factor_overlap(
    strategy_id: Annotated[str, Field(description="The node_id of the strategy to analyse for factor crowding")],
) -> str:
    """Check how many factors a strategy shares with other strategies.

    Returns a JSON object with is_crowded (bool), shared_factor_count (int),
    shared_factors (list of names), and affected_strategies (list of names).
    Use this to detect portfolio-level concentration risk.
    """
    from quantstack.knowledge.graph import KnowledgeGraph

    kg = KnowledgeGraph()
    result = kg.check_factor_overlap(strategy_id)
    return result.model_dump_json(indent=2)


@tool
def get_research_history(
    topic: Annotated[str, Field(description="The research topic or hypothesis area to search for")],
) -> str:
    """Search for past hypothesis results related to a topic.

    Returns a JSON array of HypothesisResult objects with hypothesis_id,
    hypothesis_text, test_date, outcome, result_sharpe, result_ic, and
    regime_at_test.  Use this to learn from previous experiments.
    """
    from quantstack.knowledge.graph import KnowledgeGraph

    kg = KnowledgeGraph()
    results = kg.get_research_history(topic)
    return json.dumps([r.model_dump() for r in results], indent=2, default=str)


@tool
def record_experiment(
    hypothesis: Annotated[str, Field(description="The hypothesis that was tested")],
    result_json: Annotated[str, Field(description="JSON string with outcome, sharpe, ic, and any other result metrics")],
    factors_used: Annotated[str, Field(description="Comma-separated list of factor names used in the experiment")],
    regime: Annotated[str, Field(description="Market regime during the test (e.g. trending_up, ranging, volatile)")],
) -> str:
    """Record a completed experiment in the knowledge graph.

    Creates hypothesis, result, and factor nodes with appropriate edges.
    Returns the hypothesis node_id.  Use this after every experiment so
    the system remembers what has been tried.
    """
    from quantstack.knowledge.graph import KnowledgeGraph

    result_dict = json.loads(result_json)
    factor_list = [f.strip() for f in factors_used.split(",") if f.strip()]

    kg = KnowledgeGraph()
    hypothesis_id = kg.record_experiment(hypothesis, result_dict, factor_list, regime)
    return json.dumps({"hypothesis_id": hypothesis_id, "status": "recorded"}, indent=2)
