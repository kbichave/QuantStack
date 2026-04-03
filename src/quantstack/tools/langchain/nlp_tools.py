"""NLP sentiment analysis tools for LangGraph agents."""

import json

from langchain_core.tools import tool


@tool
async def analyze_text_sentiment(text: str, method: str = "groq") -> str:
    """Analyze financial text sentiment with nuance beyond headline scoring.

    Groq method: Uses Groq LLM for nuanced multi-dimensional sentiment
    including revenue outlook, guidance direction, and management tone.
    FinBERT method: Uses HuggingFace FinBERT for fast local inference
    (requires `transformers` library).

    Args:
        text: Financial text to analyze (earnings transcript, press release,
              analyst note, SEC filing excerpt, etc.). Truncated to 2000 chars.
        method: "groq" (default, richer output) or "finbert" (local, faster)

    Returns JSON with sentiment analysis:
        sentiment: "bullish" | "bearish" | "neutral"
        confidence: 0.0-1.0
        dimensions: {revenue_outlook, guidance, management_tone}
        key_phrases: list of up to 5 key phrases
        method: "groq" | "finbert"
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
