# P10: Meta-Learning & Self-Improvement

**Objective:** Make the system improve itself — agent quality tracking, prompt optimization, strategy-of-strategies, and autonomous research prioritization.

**Scope:** learning/, graphs/agent_executor.py, graphs/supervisor/

**Depends on:** P05 (adaptive synthesis), P00 (wired learning modules)

**Effort estimate:** 2 weeks

---

## What Changes

### 10.1 Agent Decision Quality Tracking (Loop 5)
- Track per-agent: recommendation → outcome (win rate, avg P&L, Sharpe)
- `agent_quality_scores` table: `(agent_name, cycle_id, recommendation, outcome, quality_score)`
- Dashboard: which agents are making good/bad calls?
- Alert when agent win rate drops below 40%

### 10.2 Prompt Optimization (OPRO-Inspired)
- Existing: `optimization/opro_loop.py` uses Groq for prompt variants
- Extend: systematic A/B testing of prompt variants
- Process: generate variant → shadow-run for 2 weeks → compare quality → promote if better
- Focus on: trade_debater (highest impact), daily_planner, fund_manager

### 10.3 Strategy-of-Strategies
- Meta-model that selects strategy allocation based on regime, vol, correlation
- Input: current regime, strategy IC history, correlation structure
- Output: weight per active strategy
- Retrain monthly from realized performance

### 10.4 Autonomous Research Prioritization
- Current: research_queue processed FIFO
- New: priority scoring based on:
  - Expected alpha uplift (from domain knowledge)
  - Portfolio gap (underexplored asset/strategy type)
  - Recent failure mode frequency (from loss_analyzer)
  - Time since last investigation

### 10.5 Few-Shot Example Library
- CTO audit noted: zero few-shot examples in any prompt
- Build library of gold-standard agent outputs (curated from best trades)
- Auto-inject relevant examples into agent prompts based on current context
- Expected: 5-15% quality improvement

## Acceptance Criteria

1. Agent quality scores tracked and visible in dashboard
2. Prompt variants A/B tested (at least trade_debater)
3. Strategy allocation driven by meta-model (not equal-weight)
4. Research queue prioritized by expected value (not FIFO)
5. At least 3 agents have few-shot examples in prompts
