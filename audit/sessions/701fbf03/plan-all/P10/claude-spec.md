# P10 Spec: Meta-Learning & Self-Improvement

## Deliverables

### D1: Agent Decision Quality Tracking
- Per-agent quality scores: direction accuracy, magnitude accuracy, timing
- Rolling 21-day win rate computation
- Alert when win rate < 40% for 5 consecutive cycles
- Dashboard query: per-agent quality trend

### D2: Prompt A/B Testing
- Prompt variant storage (agent_name, variant_id, variant_prompt, status)
- Shadow execution: run variant alongside production, record both outputs
- Offline evaluation: compare quality scores after 14-day minimum observation
- Promotion if statistically significant improvement (p < 0.05)

### D3: Strategy-of-Strategies Meta-Allocator
- Input: regime, per-strategy IC, vol level, correlation structure
- Output: per-strategy weight (sum to 1.0, min 5% per active strategy)
- Model: L2-regularized linear regression, retrain monthly
- Integration: fund_manager reads meta-allocator weights

### D4: Research Prioritization
- Priority scoring: expected_alpha + portfolio_gap + failure_frequency + staleness
- Replace FIFO queue ordering
- Configurable scoring weights

### D5: Few-Shot Example Library
- Storage: agent_name, example_input, example_output, quality_score, regime, date
- Auto-extract from top 10% quality-scored outputs
- Selection: regime match + recency + quality threshold, max 3 per context
- Injection into agent prompts

## Dependencies
- P05 (Signal Synthesis): IC attribution data
- P00 (Wired Learning): strategy lifecycle, trade outcomes
