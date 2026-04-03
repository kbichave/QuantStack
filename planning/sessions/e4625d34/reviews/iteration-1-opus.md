# Opus Review

**Model:** claude-opus-4
**Generated:** 2026-04-02T00:00:00Z

---

# Critical Review: CrewAI Migration Implementation Plan

## 1. Completeness — Needs Improvement
- Scheduler is missing (cron jobs, community intel weekly cadence)
- Data acquisition pipeline absent (start.sh does 13+ steps)
- Parallel research workers regressed to single crew
- Options execution tools missing separate wrapper
- ~15 tool modules (intraday, NLP, attribution, feedback, learning, meta) have no wrapper mapping

## 2. Feasibility — Needs Improvement
- `asyncio.run()` in tool wrappers will fail if already inside event loop — need `nest_asyncio`
- 60-second trading cycle impossible for 11-task sequential pipeline (5-15 min realistic)
- Supervisor restarting Docker containers needs Docker socket mount (security surface)
- CrewAI async_execution is experimental

## 3. Risk — Critical Issue
- Big-bang migration of running trading system with no rollback
- CrewAI framework maturity concerns for weeks-long unattended operation
- 7 Docker services on single macOS machine — memory pressure
- Need minimum machine specs (16GB min, 32GB recommended)

## 4. Architecture — Good (with caveats)
- Hybrid sequential/parallel in TradingCrew complicates error handling
- No event bus between crews (cross-crew communication via DB not explicit)
- CrewAI memory storage backend unclear

## 5. Migration — Critical Issue
- No parallel running period / shadow mode
- No data migration verification for RAG ingestion
- No rollback plan
- No pre-deployment validation of LLM decisions

## 6. LLM-Reasoned Risk — Critical Issue
- LLM can hallucinate 40% position sizes — no backstop except broker balance
- Prompt injection via malformed RAG data could influence risk decisions
- Stochastic decisions (non-deterministic) vs current deterministic risk gate
- Daily loss halt loses crash-resistant persistence
- Risk reasoning takes 5-30s vs <1ms for current gate
- MUST add hard programmatic limits as outer boundary:
  - Max position size: 15% of equity
  - Daily loss halt: -3% automatic
  - Min liquidity: 200K ADV
  - Max gross exposure: 200%
- Use temperature 0 for risk decisions
- Use structured output (JSON) for risk decisions

## 7. Missing Details
- No new DB tables specified (crew_run_log, agent_decision_log)
- Langfuse key generation procedure missing
- Ollama model pull in Docker Compose (no init containers)
- Complete .env.example template missing
- Log management / rotation not specified
- Bedrock auth security in Docker containers

## 8. Ordering — Good
- Testing should be written alongside, not last

## 9. Testing — Needs Improvement
- No E2E smoke test for full pipeline
- No load/stress/soak testing
- Shadow mode mentioned but not deliverable
- No fallback chain test
- No graceful shutdown test

## 10. Cost — Needs Improvement
- ~4,290 LLM calls/day for trading alone at 60s cycles
- Estimated $30-50/day trading + research = $500-1000/month
- Ollama needs 6GB+ RAM (embeddings + inference)
- Langfuse storage grows rapidly — no retention policy
- 5-minute cycles would cut costs 80%

## Summary of Critical Issues

| # | Issue | Severity |
|---|-------|----------|
| 1 | No programmatic risk backstop | Critical |
| 2 | Big-bang with no rollback | Critical |
| 3 | 60s cycle impossible for 11-task pipeline | High |
| 4 | Daily loss halt loses deterministic enforcement | High |
| 5 | ~15 tool modules unmapped | Medium |
| 6 | Scheduler/data sync/preflight missing | Medium |
| 7 | No cost estimation | Medium |
| 8 | Supervisor Docker socket requirement | Medium |
