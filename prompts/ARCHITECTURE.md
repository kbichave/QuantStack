# Prompt Architecture: Ralph Loops vs Agent Orchestration

## Current Design (Sequential Ralph Loops)

```
research_loop.md (orchestrator)
├── Iteration N: Load ALL context → Decide "work on investment" → Load investment prompt → Execute Steps A-D → Write
├── Iteration N+1: Load ALL context → Decide "work on swing" → Load swing prompt → Execute Steps A-D → Write
└── Iteration N+2: Load ALL context → Decide "work on options" → Load options prompt → Execute Steps A-D → Write
    (30+ iterations to cover 10 symbols × 3 domains)
```

**Problems:**
- Sequential bottleneck (3 domains × N symbols done serially)
- Token waste (load everything to decide which domain)
- Slow convergence (rotate through domains artificially)
- Under-utilizes parallelism (domains are independent)

---

## Proposed Design: Agent-First Orchestration

### **Option 1: Pure Parallel Domain Agents (Recommended)**

```
research_orchestrator.md (lightweight decision-maker)
├── Load portfolio gaps, priority scores
├── Decide parallelism mode:
│   ├── BLITZ MODE (gaps > 30%): Spawn 3 agents per symbol in parallel
│   │   ├── equity-investment-agent (symbol=AAPL) ──┐
│   │   ├── equity-swing-agent (symbol=AAPL)        ├─→ Run in parallel
│   │   └── options-agent (symbol=AAPL)         ────┘
│   │   └── Same for NVDA, MSFT, etc.
│   │
│   └── DEEP DIVE MODE (gaps < 30%): Sequential focus on worst domain
│       └── Spawn single agent for deep research
│
└── Every 5 iterations: cross-pollination review (sequential)
```

**Each agent is self-contained:**
```
agents/equity_investment_researcher.md
├── Identity: autonomous investment researcher for ONE symbol
├── Context loading: investment-specific only (financials, transcripts, valuations)
├── Execute Steps A→B→C→D (evidence → hypothesis → strategy → validation)
├── Write results to DB + memory/tickers/{symbol}.md
└── Return summary → orchestrator aggregates
```

**Benefits:**
- 3x-10x faster convergence (domains run in parallel)
- Token efficient (each agent loads only its context)
- Natural isolation (agents can't interfere)
- Scales horizontally (add more symbols without slowdown)
- Better for Ralph loops (independent agents, not monolithic prompts)

**Tradeoffs:**
- Need cross-domain coordination layer (handled by orchestrator every 5 iterations)
- DB write conflicts (use transactions, already implemented)
- More agents = more API cost (but faster results)

---

### **Option 2: Task Queue + Worker Pool**

```
research_orchestrator.md
├── Scan gaps → Create priority queue:
│   [
│     {domain: "swing", symbol: "NVDA", priority: 0.9},
│     {domain: "investment", symbol: "AAPL", priority: 0.8},
│     {domain: "options", symbol: "AAPL", priority: 0.6},
│     ...
│   ]
│
├── Spawn N generic workers (parallel):
│   └── worker-agent (pulls task, executes domain prompt, marks complete, pulls next)
│
└── Workers run until queue empty
```

**Benefits:**
- Dynamic parallelism (spawn as many workers as token budget allows)
- Priority-based (biggest gaps first)
- Load balancing (tasks distribute across workers)

**Tradeoffs:**
- More complex state (task queue, locks)
- Harder to debug (which worker did what?)

---

### **Option 3: Hybrid - Smart Orchestrator (Easiest Migration)**

Keep current structure BUT add parallelism modes:

```python
# In research_loop.md Step 2:

completion_pct = count_met_criteria() / total_criteria

if completion_pct < 0.30:  # BLITZ MODE
    # Portfolio is sparse → maximize parallelism
    top_symbols = get_highest_priority_symbols(limit=5)
    agents = []
    for symbol in top_symbols:
        agents.append(spawn(agent="equity-investment-researcher", symbol=symbol))
        agents.append(spawn(agent="equity-swing-researcher", symbol=symbol))
        agents.append(spawn(agent="options-researcher", symbol=symbol))

    # All 15 agents (5 symbols × 3 domains) run in parallel
    results = wait_all(agents)

else:  # DEEP DIVE MODE
    # Portfolio is mature → focus on weak spots
    weakest_domain = select_domain_by_priority()  # Current logic
    # Execute sequentially (current behavior)
```

**Benefits:**
- Minimal changes to existing prompts
- 3x-15x speedup during portfolio building phase
- Still allows deep dives for fine-tuning
- Token efficient (parallel only when beneficial)

**Tradeoffs:**
- Doesn't fully exploit parallelism
- Still has rotation logic

---

## Trading Loop: Already Better Designed

The trading loop ALREADY thinks in agents:
- "Spawn 3 position-monitor agents for 3 positions" (Step 2)
- "Spawn 4 trade-debater agents for 4 candidates" (Step 3)

**But can be improved:**

```
trading_orchestrator.md
├── Step 0: Safety gate
├── Step 1: Load context (positions, watchlist, alerts)
├── Step 2: PARALLEL - Position Monitoring
│   └── Spawn position-monitor × N open positions (ALL in parallel, not serial)
├── Step 3: PARALLEL - Entry Scanning
│   └── Spawn entry-scanner × M watchlist symbols (ALL in parallel)
├── Step 4: Collect agent outputs
│   ├── Exit signals from position monitors → execute_trade(action="close")
│   └── Entry candidates from scanners → spawn trade-debater for each
├── Step 5: PARALLEL - Entry Debates
│   └── Spawn trade-debater × K candidates (ALL in parallel)
│   └── Spawn fund-manager to review batch holistically
└── Step 6: Execute approved trades (sequential for audit trail)
```

**Change:** Instead of orchestrator doing the work, spawn specialized agents for each task.

---

## Concrete Recommendation: What to Do Now

### **Phase 1: Add Parallelism Modes (1-2 hours)**

1. **Add to `research_loop.md` after "STEP 2: DECIDE WHICH DOMAIN":**

```markdown
### 2b-NEW: Decide Parallelism Mode

```python
completion_pct = len(met_criteria) / len(all_criteria)

if completion_pct < 0.30:
    mode = "BLITZ"  # Sparse portfolio → maximize parallelism
    print("BLITZ MODE: Spawning parallel domain agents")

    top_symbols = get_highest_priority_symbols(limit=3)  # Start with 3, scale up
    agents = []

    for symbol in top_symbols:
        # Spawn all 3 domains in parallel for each symbol
        agents.append({
            "agent": "quant-researcher",
            "instructions": f"Execute prompts/research_equity_investment.md for {symbol}",
            "symbol": symbol,
        })
        agents.append({
            "agent": "quant-researcher",
            "instructions": f"Execute prompts/research_equity_swing.md for {symbol}",
            "symbol": symbol,
        })
        agents.append({
            "agent": "quant-researcher",
            "instructions": f"Execute prompts/research_options.md for {symbol}",
            "symbol": symbol,
        })

    # Spawn all agents in parallel (3 domains × 3 symbols = 9 concurrent agents)
    results = [spawn_agent(**agent_config) for agent_config in agents]

    # Wait for all to complete, then aggregate results
    # (orchestrator just collects + writes cross-domain summary)

elif completion_pct < 0.70:
    mode = "DEEP_DIVE"  # Current sequential behavior
    # Pick worst domain, go deep (existing logic)

else:
    mode = "FINE_TUNE"  # Cross-pollination, portfolio optimization
```

2. **Convert domain prompts into "agent instructions":**
   - They already are mostly self-contained
   - Just need to ensure they write results properly and return summary

3. **Test with 3 symbols first**, then scale up

---

### **Phase 2: Full Agent Refactor (4-6 hours)**

1. **Move domain prompts to `agents/` directory:**
   - `agents/equity_investment_researcher.md`
   - `agents/equity_swing_researcher.md`
   - `agents/options_researcher.md`

2. **Slim down orchestrators to pure coordination:**
   - Gap analysis
   - Priority scoring
   - Agent spawning
   - Result aggregation

3. **Same for trading:**
   - `agents/position_monitor.md`
   - `agents/entry_scanner.md`
   - `agents/trade_debater.md`

---

## My Recommendation: Phase 1 First

**Do Phase 1 now** (add parallelism modes to existing structure). This gives you:
- 3x-9x speedup immediately
- Minimal risk (fall back to sequential if parallel fails)
- Proof of concept for agent architecture

**Then Phase 2** (full refactor) if Phase 1 proves the value.

The current design is "okay" but massively under-utilizes parallelism. Research could be 10x faster with proper agent orchestration.
