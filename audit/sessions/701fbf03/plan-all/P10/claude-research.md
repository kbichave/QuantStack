# P10 Research: Meta-Learning & Self-Improvement

## Codebase Research

### What Exists
- **IC Attribution**: `src/quantstack/learning/ic_attribution.py` — per-collector IC tracking, regime-aware weights
- **Strategy lifecycle**: `src/quantstack/autonomous/strategy_lifecycle.py` — promote/demote/retire pipeline
- **Feedback flags**: `src/quantstack/config/feedback_flags.py` — feature flag system for all subsystems
- **Signal synthesis**: `src/quantstack/signal_engine/synthesis.py` — weight profiles, conviction factors
- **Agent configs**: `src/quantstack/graphs/*/config/agents.yaml` — agent definitions with tools and prompts
- **Research queue**: existing research priority queue in DB

### What's Needed (Gaps)
1. **Agent quality tracking**: No per-agent quality scoring exists — need to track recommendation accuracy
2. **Prompt A/B testing**: No variant management or shadow execution — need parallel prompt evaluation
3. **Strategy-of-strategies meta-allocator**: No meta-model for strategy allocation — currently equal-weight
4. **Research prioritization scoring**: Research queue is FIFO — needs priority scoring
5. **Few-shot example library**: No curated example storage or injection — need example curation + prompt integration

## Domain Research

### Agent Quality Tracking Approaches
- Win rate (correct direction) is the primary metric, supplemented by magnitude accuracy and timing
- Rolling 21-day windows balance recency with statistical significance
- Alert threshold at 40% win rate for 5 cycles — below random chance
- Quality scores feed into few-shot selection (use outputs from top performers as examples)

### Prompt A/B Testing in LLM Systems
- Challenge: small N per prompt variant (each cycle generates 1 output per agent)
- Solution: offline evaluation — run all variants on same inputs, compare quality scores
- Statistical significance: need 50+ observations for reliable comparison (2+ weeks of daily cycles)
- Shadow execution: run variant alongside production, record both, compare after minimum observation period
