# P10 Implementation Plan: Meta-Learning & Self-Improvement

## 1. Background

P05 (adaptive synthesis) and P00 (wired learning modules) provide the data foundation: IC tracking, strategy outcomes, trade quality scores. P10 makes the system improve itself: agent quality tracking, prompt optimization, strategy-of-strategies allocation, research prioritization, and few-shot example injection.

## 2. Anti-Goals

- **Do NOT implement automated prompt rewriting** — A/B test prompt variants, don't auto-generate them
- **Do NOT replace human-curated strategy rules** — meta-model supplements, doesn't override
- **Do NOT build a general AutoML system** — focused on trading-specific quality metrics
- **Do NOT modify agent core logic** — inject improvements through prompts and configuration

## 3. Agent Decision Quality Tracking

### 3.1 Schema
`agent_quality_scores` table: (agent_name, cycle_id, recommendation_type, recommendation, outcome, quality_score, computed_at)

### 3.2 Tracking Logic
New `src/quantstack/learning/agent_quality.py`:
- After each cycle, compare agent recommendation with realized outcome
- Scoring: correct_direction (0/1), magnitude_accuracy (0-1), timing (early/on-time/late)
- Aggregate: rolling 21-day win rate, avg quality score per agent
- Alert when agent win rate drops below 40% for 5 consecutive cycles

### 3.3 Dashboard Query
`get_agent_quality_dashboard()` → per-agent win rate, quality trend, best/worst agents

## 4. Prompt A/B Testing

### 4.1 Variant Management
`prompt_variants` table: (agent_name, variant_id, variant_prompt, created_at, status)

### 4.2 Shadow Testing
- Generate 2-3 prompt variants per target agent (trade_debater, daily_planner, fund_manager)
- Shadow-run variant alongside production: same input, record both outputs
- Compare quality scores after 2 weeks minimum
- Promote if statistically significant improvement (p < 0.05)

### 4.3 Integration
In agent executor: check for active variant, run both production and shadow, record outputs

## 5. Strategy-of-Strategies

### 5.1 Meta-Model
`src/quantstack/learning/meta_allocator.py`:
- Input: current regime, per-strategy IC (21d rolling), vol level, correlation structure
- Output: weight per active strategy (sum to 1.0)
- Model: simple linear regression initially (can upgrade to ML later)
- Retrain monthly from realized performance

### 5.2 Integration
In fund_manager node: read meta-allocator weights instead of equal-weight allocation

## 6. Research Prioritization

### 6.1 Priority Scoring
Replace FIFO research_queue with priority-scored queue:
- Expected alpha uplift: domain heuristic (0-1)
- Portfolio gap: 1.0 if underexplored asset/strategy, 0.0 if saturated
- Failure frequency: normalized count from loss_analyzer
- Staleness: time since last investigation (decay function)
- Combined: weighted sum, configurable weights

### 6.2 Integration
In research graph's queue consumer: sort by priority score, process highest first

## 7. Few-Shot Example Library

### 7.1 Storage
`few_shot_examples` table: (agent_name, example_input, example_output, quality_score, created_at)

### 7.2 Curation
- Auto-extract from top 10% quality-scored agent outputs
- Manual curation interface (mark examples as gold-standard)
- Max 3 examples per agent per context type

### 7.3 Injection
In agent prompt builder: retrieve relevant examples based on current context (regime, strategy type), inject into prompt

## 8. Testing

- Agent quality: mock cycle outcomes, verify scoring
- Prompt A/B: verify shadow execution records both outputs
- Meta-allocator: train on synthetic data, verify valid weights
- Research priority: verify ordering matches expected priority
- Few-shot: verify example injection in prompt
