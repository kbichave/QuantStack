"""LLM-as-judge evaluator for post-trade quality scoring.

Uses openevals to produce structured TradeQualityScore assessments
for closed trades, consumed by the reflection node and WeightLearner.
"""

from __future__ import annotations

from typing import Any, Callable

from quantstack.performance.models import TradeQualityScore

_TRADE_QUALITY_PROMPT = """\
You are an expert trade quality evaluator for an autonomous trading system.

Evaluate the following closed trade on six dimensions, each scored 0.0-1.0:

1. **execution_quality** — Fill quality, slippage, order management.
2. **thesis_accuracy** — Did the entry thesis play out as expected?
3. **risk_management** — Was risk contained within pre-defined parameters?
4. **timing_quality** — Entry/exit timing relative to the price move.
5. **sizing_quality** — Position sizing relative to conviction level.
6. **overall_score** — Composite assessment across all dimensions.

**Entry context:**
{inputs}

**Trade outcome:**
{outputs}

Score each dimension independently. Provide a single justification string
explaining your rationale across all dimensions. Be specific about what
went well and what could improve.
"""

# Model tier for trade evaluation — medium keeps costs manageable
# since the evaluator runs on every closed trade.
_DEFAULT_MODEL = "anthropic/claude-sonnet-4-20250514"


def create_trade_evaluator(
    model: str | None = None,
) -> Callable[..., dict[str, Any]]:
    """Build an LLM-as-judge evaluator for trade quality scoring.

    Returns a callable that accepts ``inputs`` and ``outputs`` kwargs
    and returns a dict conforming to TradeQualityScore.
    """
    from openevals import create_llm_as_judge  # lazy: optional dependency

    evaluator = create_llm_as_judge(
        prompt=_TRADE_QUALITY_PROMPT,
        model=model or _DEFAULT_MODEL,
        output_schema=TradeQualityScore,
        continuous=True,
        use_reasoning=True,
    )
    return evaluator
