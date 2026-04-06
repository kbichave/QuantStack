"""NLP sentiment analysis tools for LangGraph agents."""

import json
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field


@tool
async def analyze_text_sentiment(
    text: Annotated[str, Field(description="Financial text to analyze for sentiment: earnings transcript, press release, analyst note, SEC filing excerpt, or news article. Truncated to 2000 chars")],
    method: Annotated[str, Field(description="Sentiment analysis backend: 'groq' for LLM-based multi-dimensional analysis or 'finbert' for fast local HuggingFace FinBERT inference")] = "groq",
) -> str:
    """Analyzes financial text sentiment using NLP to classify tone as bullish, bearish, or neutral with confidence scoring. Use when evaluating earnings call transcripts, press releases, analyst notes, SEC filings, or news headlines for trading signal generation. Computes multi-dimensional sentiment including revenue outlook, forward guidance direction, and management tone via Groq LLM or local FinBERT model. Returns JSON with sentiment label, confidence score (0-1), dimensional breakdown (revenue_outlook, guidance, management_tone), up to 5 key extracted phrases, and the method used."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
