# Section 08: Financial NLP (FinBERT Integration)

## Objective

Add FinBERT-based financial sentiment analysis alongside the existing LLM-based sentiment collector. FinBERT is a domain-specific BERT model pre-trained on financial text — it provides more reliable sentiment scores for financial language than general-purpose LLMs.

## Files to Create/Modify

### New Files

- **`src/quantstack/ml/nlp/__init__.py`** — Package init.
- **`src/quantstack/ml/nlp/financial_sentiment.py`** — FinBERT inference wrapper.

### Modified Files

- **`src/quantstack/signal_engine/collectors/sentiment.py`** — Add FinBERT score alongside existing Groq/LLM sentiment; ensemble the two.

## Implementation Details

### `src/quantstack/ml/nlp/financial_sentiment.py`

```
class FinBERTSentiment:
    """FinBERT-based financial sentiment scorer.
    
    Uses ProsusAI/finbert from HuggingFace — pre-trained on financial news,
    analyst reports, and SEC filings. No fine-tuning needed.
    
    Outputs sentiment in [-1.0, 1.0]: negative = bearish, 0 = neutral, positive = bullish.
    """

    _instance: "FinBERTSentiment | None" = None  # singleton for model reuse

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = "cpu"):
        """Load FinBERT model and tokenizer.
        
        Uses HuggingFace transformers pipeline.
        Model is ~440MB — loaded once and cached in memory (singleton pattern).
        """

    @classmethod
    def get_instance(cls) -> "FinBERTSentiment":
        """Get or create singleton instance (avoids reloading 440MB model)."""

    def score_text(self, text: str) -> FinBERTScore:
        """Score a single text string.
        
        Returns FinBERTScore with sentiment value in [-1, 1] and confidence.
        Maps FinBERT's 3-class output (positive/negative/neutral) to a continuous score:
            score = P(positive) - P(negative)
            confidence = max(P(positive), P(negative), P(neutral))
        """

    def score_headlines(self, headlines: list[str]) -> FinBERTScore:
        """Score multiple headlines and return aggregate.
        
        Aggregation: weighted mean by confidence (higher-confidence headlines count more).
        """
```

```
@dataclass
class FinBERTScore:
    sentiment: float      # [-1.0, 1.0]
    confidence: float     # [0.0, 1.0]
    label: str            # "positive", "negative", "neutral"
```

### Sentiment Collector Modification

Modify `collect_sentiment()` in `sentiment.py`:

1. After getting the existing Groq/LLM sentiment score, also run FinBERT on the same headlines.
2. Ensemble: `final_score = 0.6 * finbert_score + 0.4 * llm_score` (FinBERT weighted higher — more reliable for financial text).
3. Add new output keys:
   - `sentiment_finbert`: float — raw FinBERT score [-1, 1], mapped to [0, 1] for compatibility
   - `sentiment_llm`: float — original LLM score (renamed from `sentiment_score` for clarity)
   - `sentiment_score`: float — ensembled score (maintains backwards compatibility)
   - `sentiment_source`: str — "ensemble" (or "llm_only" / "finbert_only" if one source fails)
4. If FinBERT fails (model load error, OOM), fall back to LLM-only gracefully.
5. Track IC separately for each source to evaluate which provides more alpha over time.

### Key Design Decisions

1. **ProsusAI/finbert** — most widely used financial sentiment model, well-validated.
2. **No fine-tuning** — the plan explicitly excludes local LLM fine-tuning. Use pre-trained weights.
3. **Singleton pattern** — 440MB model loaded once, reused across all symbols in a cycle.
4. **0.6/0.4 ensemble** — FinBERT higher weight because it is specifically trained on financial language; the LLM adds breadth for non-standard text.
5. **Backwards compatible** — `sentiment_score` key is preserved (now ensembled instead of LLM-only).

## Dependencies

- **PyPI**: `transformers`, `torch` (HuggingFace transformers for FinBERT inference)
- **Internal**: `quantstack.signal_engine.collectors.sentiment` (existing collector)

## Test Requirements

### `tests/unit/ml/test_financial_nlp.py`

1. **Valid score range**: FinBERT output is always in [-1.0, 1.0].
2. **Positive text**: "Company reports record earnings, stock surges" -> positive sentiment.
3. **Negative text**: "Company misses earnings, guidance lowered significantly" -> negative sentiment.
4. **Neutral text**: "Company schedules annual shareholder meeting" -> near-zero sentiment.
5. **Empty input**: Empty string or empty headline list returns neutral score (0.0).
6. **Singleton**: Two calls to `get_instance()` return the same object.
7. **Headline aggregation**: Multiple headlines with mixed sentiment produce a weighted average.

### `tests/unit/signal_engine/test_sentiment_ensemble.py`

1. **Ensemble math**: With FinBERT=0.8 and LLM=0.6, ensemble = 0.6*0.8 + 0.4*0.6 = 0.72.
2. **FinBERT failure fallback**: When FinBERT raises, collector returns LLM-only score with source="llm_only".
3. **Both sources present**: Output includes `sentiment_finbert`, `sentiment_llm`, and `sentiment_score`.
4. **Backwards compatibility**: `sentiment_score` key always present in output dict.

## Acceptance Criteria

- [ ] FinBERT loads and produces sentiment scores in [-1, 1] for financial text
- [ ] Singleton pattern prevents redundant model loading
- [ ] Sentiment collector ensembles FinBERT + LLM with 0.6/0.4 weighting
- [ ] Backwards compatible — `sentiment_score` key preserved
- [ ] Graceful fallback when FinBERT is unavailable (LLM-only mode)
- [ ] IC tracked separately per source for future weight adjustment
- [ ] All unit tests pass
- [ ] CPU inference only — no GPU required
