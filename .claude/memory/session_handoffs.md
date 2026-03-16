# Session Handoffs

> Cross-session context + self-modification log
> Read at start of: every session
> Update: when context transfers needed, when any config/skill file modified

## Handoff Log

| Date | From Session | To Session | Key Context |
|------|-------------|------------|-------------|
| 2026-03-15 | config | all | Local Ollama set as primary LLM provider. See self-modification log below. |

---

## Self-Modification Log

### 2026-03-15 — Local Ollama setup (M3 Max 128GB)

**Files modified:**
- `packages/quant_pod/llm_config.py` — Added `workshop` tier, tier-aware `_model_ollama`, `_build_ollama_llm` helper (injects `api_base` + `extra_body={"think": False}`), `get_llm_for_role()` public function. `get_llm_for_agent` now returns `crewai.LLM` object for Ollama models instead of plain string.
- `.env` — Switched `LLM_PROVIDER=ollama`. Set per-tier overrides: IC/decoder → qwen3.5:9b, pod/assistant → qwen3.5:35b-a3b. Workshop → bedrock/claude-sonnet-4. Fallback chain: bedrock,openai.
- `.env.example` — Added Ollama section with comments.
- `scripts/check_ollama_health.py` — New script (stdlib only). Checks server reachability, pulled models, loaded models, auto-preloads if needed. Exit 0 = healthy.
- `.claude/skills/trade.md` — Step 0 now runs health check first; abort if models not loaded.
- `.claude/skills/workshop.md` — Step 0 now runs health check + AWS credential verify.
- `CLAUDE.md` — LLM Configuration section replaced with local model tables, cost breakdown, health check instructions, launchd startup config.

**Why:** M3 Max has 128GB unified memory. Two models (~36GB peak total) run permanently at 0 cost vs ~$0.03/crew run on Bedrock. NUM_PARALLEL=10 enables all 10 ICs to hit qwen3.5:9b simultaneously. Thinking mode disabled for agents via `extra_body={"think": False}` — saves tokens/latency with no quality loss for focused IC work. Workshop sessions keep Bedrock Sonnet for deep reasoning quality.

**Still needed (manual steps — Ollama not running at time of config):**
1. `ollama serve` or start Ollama app
2. `ollama pull qwen3.5:9b` and `ollama pull qwen3.5:35b-a3b`
3. Set launchd env vars: KEEP_ALIVE=-1, FLASH_ATTENTION=1, NUM_PARALLEL=10, then restart Ollama
4. Run `python scripts/check_ollama_health.py` to verify
(empty)

## Self-Modification Log

| Date | File Changed | What Changed | Why | Evidence |
|------|-------------|-------------|-----|----------|
| 2026-03-15 | (initial creation) | All config/memory/skill files | Phase 1 post-config setup | First-time system initialization |
| 2026-03-15 | packages/quant_pod/mcp/server.py | Added IC output cache (_ic_output_cache, _ic_cache_set, _ic_cache_get, _populate_ic_cache_from_result) + 7 new MCP tools (list_ics, run_ic, run_pod, run_crew_subset, get_last_ic_output, get_fill_quality, get_position_monitor) | Enhancement 1 (Granular IC Access) + Enhancement 5 (Execution Feedback Loop) | 19 tests passing |
| 2026-03-15 | .claude/skills/deep_analysis.md | Created new reference skill for QuantCore tool usage | Enhancement 2 (QuantCore integration) | New file, covers pre-trade/signal/risk enrichment categories |
| 2026-03-15 | .claude/skills/trade.md | Added Steps 3.5 (targeted pre-screen), 4a (pre-trade intelligence), 4b (signal enrichment), 4c (risk enrichment) | Enhancements 2+3 (QuantCore integration + chained analysis) | Skills-based conditional logic chosen over MCP chain tool |
| 2026-03-15 | .claude/skills/meta.md | Added Step 5a (cross-strategy correlation check) | Enhancement 2 (QuantCore integration for portfolio-level analysis) | Between multi-symbol analysis and conflict resolution |
| 2026-03-15 | .claude/skills/review.md | Added Step 2a (position monitor via get_position_monitor), Step 9 (fill quality audit) | Enhancement 5 (execution feedback loop) | Weekly fill audit covers last 20 fills |
| 2026-03-15 | scripts/scheduler.py | Created APScheduler-based session scheduler daemon | Enhancement 4 (scheduled session triggers) | 4 jobs: 9:15/12:30/15:45 Mon-Fri + 17:00 Fri; --dry-run/--cron/--run-now CLI flags |
| 2026-03-15 | pyproject.toml | Added scheduler optional dependency (apscheduler>=3.10.0) | Enhancement 4 support | Optional install, not required by core system |
| 2026-03-15 | CLAUDE.md | Added E1 (Granular IC Access) and E5 (Execution Feedback Loop) tool tables; added deep_analysis.md reference; added Automated Session Triggers table | Documentation for all 5 enhancements | Final CLAUDE.md reflects full Phase 1-6 tool inventory |
| 2026-03-15 | tests/quant_pod/test_ic_access.py | Created 19 tests for E1+E5 enhancements | Test coverage for new MCP tools | All 19 passing; parameter order fix applied to get_last_ic_output |
