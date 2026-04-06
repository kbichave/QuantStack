"""Cross-domain intelligence tools for LangGraph agents."""

import json
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field


@tool
async def get_cross_domain_intel(
    symbol: Annotated[str, Field(description="Filter to a specific ticker symbol. Empty string returns intel for all active symbols")] = "",
    requesting_domain: Annotated[str, Field(description="The requesting research domain: 'equity_investment', 'equity_swing', 'options', or empty for all domains")] = "",
    include_stale: Annotated[bool, Field(description="Whether to include intelligence from alerts older than 14 days")] = False,
) -> str:
    """Retrieve cross-domain intelligence that surfaces signals and artifacts from other research domains relevant to the requesting domain. Use when a research or trading agent needs context from sibling domains, such as options flow informing equity swing decisions or investment thesis supporting options trades. Returns JSON with intel_items sorted by relevance, a summary narrative, and symbol_convergence showing multi-domain agreement. Synonyms: multi-domain signals, cross-strategy intel, domain convergence, signal aggregation, inter-domain context, strategy overlap."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
