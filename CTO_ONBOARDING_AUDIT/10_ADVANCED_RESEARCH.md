# 09 — Advanced Research Integration: From Trading System to Trading Company

**Priority:** P4 — Scale and Self-Improve
**Timeline:** Week 8+
**Gate:** Phase 1-3 from Sections 01-06 complete. System profitable in paper trading with validated signals.

---

## Why This Section Is Last

These are visionary capabilities — autoresearch loops, metacognitive self-modification, knowledge graphs, hierarchical governance. They are the right destination. But building them on unvalidated signals and phantom execution would be building a self-improving machine that improves in the wrong direction.

**Prerequisite:** Sections 01-06 must be substantially complete before starting this work.

---

## The Core Insight

QuantStack already has the bones of a frontier system. What's missing is the *meta layer* — the system doesn't improve *how it improves*. It discovers strategies, but it doesn't discover better ways to discover strategies. It fixes bugs, but it doesn't fix the patterns that cause bugs.

Eight frontier AI research systems were reviewed: Karpathy's autoresearch, DGM-Hyperagents, Mimosa, Kosmos, ResearchGym, MAGNET, AI-Supervisor, and OrgAgent. Their architectures map to concrete upgrades for QuantStack.

---

## AR-1: Overnight Autoresearch Loop (Karpathy Pattern)

**Effort:** 1 week | **Impact:** 16x research throughput

### The Concept

Fixed 5-minute experiment budgets. ~96 experiments per night. Single metric (OOS IC). Haiku for hypothesis generation. Discard failures immediately. Keep winners for morning validation.

```
OVERNIGHT MODE (20:00 → 04:00 ET, 8 hours):

program.md (human-written strategy themes, constraints)
  → FIXED 5-MIN EXPERIMENT:
    1. Generate hypothesis (30s, Haiku)
    2. Compute features (60s, deterministic)
    3. Train LightGBM draft (120s, CPU)
    4. Measure OOS IC + Sharpe (30s, deterministic)
    5. Keep if IC > prev_best, else discard
  → ~96 experiments per night
  → Winners → morning strategy pipeline
```

### Implementation

| Step | Action |
|------|--------|
| 1 | Add `overnight_mode` to research graph runner |
| 2 | Create `autoresearch_node.py` with fixed 5-min budget |
| 3 | Use Haiku for hypothesis generation — speed > depth |
| 4 | Single metric: OOS IC on purged holdout |
| 5 | Store log in `autoresearch_experiments` table |
| 6 | Morning pipeline evaluates winners with full backtest |

---

## AR-2: Metacognitive Self-Modification (DGM-Hyperagents)

**Effort:** 3 weeks | **Impact:** Compounding quality improvement

### The Concept

Meta agents improve task agents. They edit prompts, modify tool selections, adjust thresholds, and improve their own modification procedure.

### Four Meta Agents

| Agent | What It Does | Frequency |
|-------|-------------|-----------|
| `meta_prompt_optimizer` | Analyze trade outcomes per agent. If win rate drops, generate prompt variant. A/B test. | Weekly |
| `meta_threshold_tuner` | Is 0.7 confidence threshold too lenient or strict? Adjust by 0.05 increments. | Monthly |
| `meta_tool_selector` | Track tool invocation patterns. Inject unused-but-needed tools. Remove always-failing tools. | Weekly |
| `meta_architecture_critic` | Compare realized Sharpe to benchmark. Identify bottleneck subsystem. Propose improvement. | Quarterly |

**The recursive piece:** `meta_prompt_optimizer` itself has a prompt. If its suggestions don't improve performance, `meta_architecture_critic` modifies the optimizer's prompt.

---

## AR-3: Alpha Knowledge Graph (AI-Supervisor Pattern)

**Effort:** 4 weeks | **Impact:** Institutional memory, factor crowding prevention

### The Concept

Structured knowledge graph of factors, strategies, hypotheses, results, and relationships. Agents query it to avoid redundant work and identify unexplored territory.

### Schema

```
Nodes: Factor, Strategy, Hypothesis, Result, Evidence, Regime
Edges: FACTOR --uses--> STRATEGY
       HYPOTHESIS --tested_by--> BACKTEST
       STRATEGY --correlates_with--> STRATEGY (>0.6 corr)
       FACTOR --contradicted_by--> PAPER
       REGIME --favors--> FACTOR
```

### Killer Feature

When `fund_manager` evaluates a new entry: "Does this strategy share >2 factors with any existing position?" If yes, apply correlation haircut. This catches *factor* correlation (smarter than price correlation).

### Implementation

Use PostgreSQL JSON columns, not a separate graph DB. Add `alpha_knowledge_graph` table with node/edge schema.

---

## AR-4: Hierarchical Governance (OrgAgent Pattern)

**Effort:** 3 weeks | **Impact:** 74% token reduction, same or better quality

### The Concept

Separate governance (decides what), execution (does it), and compliance (audits it).

```
GOVERNANCE (CIO Agent) — runs once per morning, Sonnet
  Sets: capital allocation per strategy, max new positions, regime stance
      ↓ mandates
EXECUTION (Strategy Agents) — run every 5 min, Haiku
  momentum_agent, mean_rev_agent, options_agent, earnings_agent
  Each: proposes trades within mandate
      ↓ proposals
COMPLIANCE (Risk Officer) — per-trade, mostly deterministic
  Validates against risk_gate + knowledge graph + mandate
  IMMUTABLE — same protection as risk_gate.py
```

**Token savings:** CIO at $0.15/day + 4 strategy agents at Haiku (~$3.20/day) + deterministic compliance = ~$10/day vs. current $150-450/day.

---

## AR-5: Long-Horizon Parallel Research (Kosmos Pattern)

**Effort:** 2 weeks | **Impact:** 224x weekend research throughput

### The Concept

4 parallel research streams running Friday 20:00 → Monday 04:00 (56 hours):

| Stream | Focus |
|--------|-------|
| Factor Mining | Academic papers → testable factors → IC screening |
| Regime Research | Historical regime labeling → regime-conditional allocation model |
| Cross-Asset Signals | Bond-equity-FX-commodity lead-lag relationships |
| Portfolio Construction | Alternative optimizers (risk parity, Black-Litterman, HRP) |

At autoresearch speed: **~2,688 experiments per weekend** vs. ~12 today.

---

## AR-6: Consensus-Based Signal Validation

**Effort:** 1 week | **Impact:** Fewer catastrophic trades

### The Concept

For high-stakes decisions (>$5K position size): 3 independent agents (bull advocate, bear advocate, neutral arbiter) evaluate in parallel. Deterministic consensus merge:
- 3/3 agree: full size
- 2/3 agree: 50% size
- <2/3: reject

For low-stakes (<$5K): single agent (current behavior, faster).

---

## AR-7: Error-Driven Research Iteration (MAGNET + ResearchGym)

**Effort:** 1 week | **Impact:** Systematic loss reduction

### The Concept

Daily loss analysis pipeline (16:30 ET):

```
COLLECT all losing trades today
  → CLASSIFY by failure mode (regime_mismatch, factor_crowding, data_stale, etc.)
  → AGGREGATE over trailing 30 days
  → PRIORITIZE by cumulative P&L impact
  → GENERATE research tasks targeting top failure modes
  → FEED research_queue with priority = f(cumulative_loss)
```

**This closes the learning loop:** loss → classify → research → new strategy/hedge → prevented future loss.

---

## AR-8: Dynamic Tool Lifecycle (Mimosa Pattern)

**Effort:** 1 week | **Impact:** Self-improving tool layer

### The Concept

| Phase | Action |
|-------|--------|
| **Registry Cleanup** | Split TOOL_REGISTRY into ACTIVE_TOOLS (30) and PLANNED_TOOLS (92 stubs). Agents only see active. |
| **Health Monitoring** | Track invocation count, success rate, latency per tool. Disable <50% success rate. |
| **Tool Synthesis** | AutoResearchClaw implements highest-priority planned tools based on demand signals. |
| **Capability Announcement** | `TOOL_ADDED` event → agents reload bindings next cycle. |

---

## AR-9: Fixed-Budget Experiment Discipline (ResearchGym)

**Effort:** 1 week | **Impact:** Predictable costs, no runaway agents

### The Concept

| Budget Type | Limit |
|------------|-------|
| Per-cycle token budget | 50K (research), 30K (trading) |
| Per-cycle wall-clock | 600s (research), 300s (trading) |
| Per-cycle cost | $0.50 (research), $0.20 (trading) |

Prioritization formula: `priority = (expected_IC * regime_fit * novelty_score) / estimated_compute_cost`

Patience protocol: never discard after 1 backtest — run 3 time windows. Marginal IC (0.01-0.02) = test with different features before discarding.

---

## AR-10: Autonomous Feature Factory (MAGNET Pattern)

**Effort:** 2 weeks | **Impact:** 10x feature coverage

### The Concept

| Phase | What Happens |
|-------|-------------|
| **Enumeration** (weekly overnight) | For each base feature: generate lags, rolling stats, cross-interactions, regime-conditional variants, residualized features. ~500+ candidates. |
| **Screening** (weekly overnight) | Compute IC for all candidates. Filter IC > 0.01 AND stability > 0.5. Drop features with >0.95 correlation to kept features. Output: ~50-100 curated features. |
| **Monitoring** (daily) | PSI drift per feature. IC decay tracking. Auto-replace decaying features. |

---

## Implementation Roadmap

| Phase | Techniques | Timeline | Prerequisites |
|-------|-----------|----------|---------------|
| **Foundation** | Registry cleanup (AR-8), Error-driven iteration (AR-7), Experiment budgets (AR-9) | Week 8-9 | Sections 01-03 complete |
| **Research Multiplier** | Autoresearch loop (AR-1), Feature factory (AR-10), Weekend parallel research (AR-5) | Week 9-12 | Foundation phase |
| **Intelligence Layer** | Knowledge graph (AR-3), Consensus validation (AR-6) | Week 12-14 | Research output to populate graph |
| **Self-Improvement** | Meta layer (AR-2), Hierarchical governance (AR-4) | Week 14-18 | Performance data for meta-optimization |

---

## Cost-Benefit Summary

| Technique | Effort | Annual Token Savings | Alpha Impact |
|-----------|--------|---------------------|--------------|
| Autoresearch (AR-1) | 1 week | -$2K (more research) | 16x hypothesis throughput |
| Meta layer (AR-2) | 3 weeks | +$5K (meta agents) | Compounding quality improvement |
| Knowledge graph (AR-3) | 4 weeks | $0 (deterministic) | Prevent redundant research + factor crowding |
| OrgAgent hierarchy (AR-4) | 3 weeks | **-$50K** (74% token reduction) | Same or better quality |
| Consensus (AR-6) | 1 week | +$3K (3x LLM on big trades) | Fewer catastrophic trades |
| Error-driven (AR-7) | 1 week | $0 | Systematic loss reduction |
| Feature factory (AR-10) | 2 weeks | $0 (CPU compute) | 10x feature coverage |
| Experiment budgets (AR-9) | 1 week | **-$15K** (bounded spend) | Predictable costs |
| Dynamic tools (AR-8) | 1 week | $0 | Self-improving tool layer |
| Parallel research (AR-5) | 2 weeks | +$5K (weekend compute) | 224x weekend throughput |
| **NET** | **~20 weeks** | **~$54K/year savings** | **Compounding alpha + self-improvement** |

---

## AutoResearchClaw: Under-Utilized Foundation

**Source:** CTO Audit — AutoResearchClaw section (4 MEDIUM findings)

AutoResearchClaw (`scripts/autoresclaw_runner.py`, 683 lines) is a sophisticated self-healing system with production-grade safety rails (protected files, low-confidence revert, syntax validation, audit trail). But it's under-utilized:

| Gap | Current | Required | Fix |
|-----|---------|----------|-----|
| **Weekly schedule** | Sunday 20:00 only — 3 tasks/week | Nightly — 96 experiments/night (AR-1) | Change scheduler to nightly 20:00 |
| **Reactive only** | Only on failure (bug_fix) or drift | Proactive gap detection + hypothesis generation | Add automated gap detection feeds |
| **tmux restarts** | `tmux send-keys` to restart loops — fragile | Docker Compose restart or SIGHUP | Use `docker compose restart` |
| **No functional validation** | py_compile + import only | Run fixed tool with test input after fix | Add tool invocation test after patch |

AutoResearchClaw is the backbone of AR-1 (overnight autoresearch), AR-7 (error-driven iteration), and AR-8 (dynamic tool synthesis). These advanced techniques build on its existing infrastructure.

---

## The Endgame

```
TODAY (v1.0):
  Human writes strategy → System backtests → System trades → Human reviews

PHASE 2 (v2.0):
  System generates 96 hypotheses/night → Filters → Trades → Logs losses

PHASE 3 (v3.0):
  Knowledge graph prevents redundancy → Consensus validates → Loss analyzer feeds research

PHASE 4 (v4.0):
  Meta layer improves hypothesis quality → Tunes risk thresholds from outcomes →
  Optimizes its own optimization process → Compounds intelligence with capital
```

**The v4.0 system doesn't just make money — it gets better at making money, and it gets better at getting better.**
