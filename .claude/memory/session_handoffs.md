# Session Handoffs

> Cross-session context + self-modification log
> Read at start of: every session
> Update: when context transfers needed, when any config/skill file modified

## Handoff Log

| Date | From Session | To Session | Key Context |
|------|-------------|------------|-------------|
| 2026-03-15 | config | all | Local Ollama set as primary LLM provider. See self-modification log below. |
| 2026-03-15 | /workshop v4 | /meta, /trade | quality_rsimr_15d (strat_3651b3242b6e) promoted to forward_testing. Universe: XOM, IBM, MSFT. Deploy via run_multi_analysis on these 3 symbols. WF inconclusive (signal sparsity). Next: v5 options convexity or ML SHAP discovery. |
| 2026-03-15 | /workshop v5 | /meta, /trade | MTF strategies deployed. mtf_swing_rsimr (strat_6af32bf683f6) for XOM+MSFT (Sharpe 3.70/1.17). mtf_medium_rsimr (strat_e01e2da6d772) for MSFT only (Sharpe 0.96, 142 trades). IBM stays on v4 (quality_rsimr_15d). WF still inconclusive — needs run_walkforward_mtf tool. |
| 2026-03-15 | /workshop v5 | all | Infra fixes: (1) run_backtest_mtf MCP tool added to quantpod. (2) QuantPod stale lock auto-recovery in db.py + server.py. (3) QuantCore DataStore switched to short-lived connections — no more permanent write locks blocking external processes. (4) Alpaca MCP not configured yet — add to settings.json for sub-hourly data. |
| 2026-03-16 | /workshop v6 | /trade, /meta | Options strategy deployed: `options_atm_rsimr` (strat_1e4e9ba80af7) on XOM+MSFT. When RSI<35 fires, prefer ATM call 30 DTE over equity. No IV filter needed. XOM Sharpe 2.08 (47.5% avg premium return), MSFT Sharpe 3.25 (70.7%). IBM stays equity-only (v4). |
| 2026-03-16 | /workshop v6 | all | Bug fixes: (1) `save_ohlcv` column mismatch — Alpaca returns 8 cols, save expected 6. Fixed: drop non-core cols. (2) `update_strategy` calling FunctionTool as callable — fixed to `.fn()`, needs MCP restart. (3) XOM/MSFT daily+1H data now persisted in DuckDB via Alpaca. (4) 15min data confirmed available — Profile C day trade unblocked. |
| 2026-03-16 | /workshop v7 | /workshop v8 | RSI>65 overbought short FAILED: equity shorts Sharpe 0.007/-0.39/-0.50, puts Sharpe -0.14/0.30. Overbought≠oversold — positive drift means 70-80% of entries keep rising. Bidirectional MR on individual stocks is disproven. |
| 2026-03-16 | /workshop v7 | all | Bug fixes: (1) BacktestEngine now includes `entry_date`/`exit_date` in trade records (engine.py). (2) `run_backtest` now returns `trades` list in response (backtesting.py). (3) Signal generator prerequisite direction: added `parameters.direction` check to route structured entries to `entry_short` when direction=SHORT (backtesting.py). All 3 need MCP server restart to take effect. |
| 2026-03-17 | /lit-review | /workshop, all | First full literature review completed. Created `/lit-review` skill (.claude/skills/lit_review.md) + memory file (lit_review_findings.md). **P0 gaps:** (1) Grammar-guided alpha discovery (GP+RL) — AlphaCFG/HARLA outperform grid search. (2) Proactive concept drift detection — Proceed/DynaME frameworks. **P1 gaps:** Decision Transformer for meta-RL, causal discovery for alpha factors, LLM-RAG sentiment on earnings calls. See lit_review_findings.md for full roadmap. |

| 2026-03-17 | v0.5.0 build | all | Multi-strategy architecture complete. Three trading styles now supported: /trade (equity swing), /invest (fundamental, weekly), /options (event/IV-driven). See self-modification log for all files changed. |

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

### 2026-03-17 — v0.5.0 Multi-Strategy Architecture

**Files modified:**

Skills:
- `.claude/skills/tune.md` — **DELETED** (CrewAI IC tuning deprecated as of v0.3.0; fix collector code instead)
- `.claude/skills/reflect.md` — Step 4.5 updated: replaced /tune flag with collector code fix path
- `.claude/skills/deep_analysis.md` — Options Intelligence section: distinguish synthetic (`compute_option_chain`) vs live (`get_options_chain`) chain tools
- `.claude/skills/workshop.md` — Added instrument_type param section (equity/options/investment)
- `.claude/skills/review.md` — Added options position checks (DTE < 21, theta bleed, skew, 50% profit exit)
- `.claude/skills/README.md` — Removed tune, added invest + options to user-invocable list; updated session flows
- `.claude/skills/invest.md` — **NEW** — Long-term fundamental investing: FCF/quality scorecard, DCF MOS, weekly cadence
- `.claude/skills/options.md` — **NEW** — Short-term options: event calendar + IV rank decision matrix, defined-risk only

Infrastructure:
- `packages/quantcore/data/base.py` — Added `fetch_options_chain()` optional interface method
- `packages/quantcore/data/adapters/alpaca.py` — Implemented `fetch_options_chain()` via OptionHistoricalDataClient
- `packages/quantcore/data/adapters/polygon_adapter.py` — Implemented `fetch_options_chain()` via Polygon v3 snapshot endpoint
- `packages/quantcore/mcp/tools/options.py` — Added `get_options_chain()` and `get_iv_surface()` MCP tools (Alpaca → Polygon → synthetic fallback)
- `packages/quant_pod/db.py` — Added `instrument_type`, `time_horizon`, `holding_period_days` columns to strategies table
- `packages/quant_pod/mcp/tools/strategy.py` — Extended `register_strategy()` and `update_strategy()` with new fields; `get_strategy()` returns them
- `packages/quant_pod/execution/risk_gate.py` — Added `max_premium_at_risk_pct`, `max_total_premium_pct`, `min_dte_entry`, `max_dte_entry` to `RiskLimits`; `check()` now accepts `instrument_type`, `premium_at_risk`, `dte` — options get separate check path
- `CLAUDE.md` — Removed /tune from Section 11; added /invest + /options; added options risk limits to Section 8; updated register_strategy signature

**Why:** v0.4.0 was single-instrument, single-time-horizon. v0.5.0 enables three qualitatively different trading styles. No provider switch needed — FinancialDatasets.ai already integrated. Key invariant: `compute_option_chain` (synthetic) untouched — it remains the backtest tool. `get_options_chain` (live) is additive for execution only.

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
| 2026-03-15 | packages/quant_pod/tools/mcp_bridge.py | Added EmptyInput + MarketRegimeSnapshotInput schemas; added args_schema to ListStoredSymbolsTool, ListAvailableIndicatorsTool, GetMarketRegimeSnapshotTool | Groq rejects tool schemas with `required` but no `properties` | ICs were failing with schema validation error |
| 2026-03-15 | packages/quantcore/mcp/server.py | Replaced hardcoded AlphaVantageClient in fetch_market_data with DataProviderRegistry; added data_registry to ServerContext lifespan init | fetch_market_data was hardcoded to Alpha Vantage; user's preferred provider is Alpaca | Alpha Vantage key was empty; Alpaca key configured and working |
| 2026-03-15 | packages/quant_pod/mcp/server.py | Rewrote _fetch_price_data to use DataProviderRegistry fallback when DuckDB cache misses | Backtest tool couldn't find SPY data (DuckDB empty, no provider fallback) | run_backtest returned "No price data available for SPY" |
| 2026-03-15 | packages/quantcore/mcp/server.py | Removed 3 unused tools: fit_vol_surface, execute_paper_trade, get_order_book_snapshot (tool count 47→44) | User confirmed these were not used — reduce surface area | Audit found no callers in codebase |
| 2026-03-15 | CLAUDE.md | Added Sections 12 (Production Operations Stack) + 13 (Broker Integrations); updated IC/pod counts throughout | Full capability audit revealed 28+ undocumented capabilities; monitoring, guardrails, flows, API server, Alpaca/IBKR MCPs were all invisible to Claude | Capability audit agent returned 31 undocumented items |
| 2026-03-15 | .claude/skills/trade.md | Fixed IC count in Step 4: "10 ICs, 5 Pod Managers" → "13 ICs, 6 Pod Managers" | Stale count from before fundamentals_ic/news_sentiment_ic/options_flow_ic and alpha_signals_pod_manager were added | Same audit finding |
| 2026-03-15 | .claude/skills/review.md | Added intraday_monitor_flow check to Step 1 | IntradayMonitorFlow exists but /review skill had no awareness of it; regime reversals could be missed between sessions | Same audit finding |
| 2026-03-15 | .claude/skills/reflect.md | Added Step 3.5 (AlphaMonitor Discord alert review + DegradationDetector findings); renumbered ML health to Step 3.6 | AlphaMonitor and DegradationDetector both emit alerts; /reflect had no process to check them | Same audit finding |
| 2026-03-15 | packages/quant_pod/mcp/server.py | Enhanced _generate_signals_from_rules + _evaluate_rule: added ADX/CCI/StochK/price_vs_sma200/regime/ATR_pct indicators; prerequisite/confirmation rule hierarchy; position-aware exit simulation (time stops, ATR SL/TP) | Backtest engine silently ignored custom rules — only RSI/SMA/BBands/zscore worked | 12 existing tests pass; v1 Sharpe corrected from 0.178 to -0.07 |
| 2026-03-15 | .claude/memory/*.md | Workshop v2 session: regime gate disproven, SPY mean-reversion exhausted, corrected v1 results | 10+ variant backtests; best Sharpe 0.36 (below 1.0 target) | Strategy strat_493cb5448197 marked failed |
| 2026-03-15 | .claude/settings.json | Added register_strategy PostToolUse hook → log_decision.py | Strategy registrations were not being logged to audit trail; only trades were | Same audit finding |
| 2026-03-15 | .claude/memory/ml_model_registry.md | Created with model types, when-to-train guidance, status values | ML stack (ModelTrainer/HierarchicalEnsemble/HMM/TFT/Changepoint) existed with no memory tracking | ML capability audit |
| 2026-03-15 | .claude/memory/workshop_lessons.md | Added v3 multi-stock results (13 symbols, avg Sharpe -0.01, FAIL) + updated failed hypotheses table | v3 workshop session completed | 117 trades, 2/13 symbols viable |
| 2026-03-15 | .claude/memory/strategy_registry.md | Added strat_be0a6ddaf86b (multi_stock_rsimr_15d) to failed table | v3 strategy marked failed | Portfolio-level Sharpe ≈ 0 |
| 2026-03-15 | packages/quant_pod/crews/trading_crew.py | Fixed options_flow + put_call_ratio TOOL_REGISTRY bug (lambda: Class → lambda: Class()); added memory tools to registry; added strategy_context+session_notes to run_analysis_only | options_flow_ic tools were class objects not instances; memory tools unreachable | Determinism audit |
| 2026-03-15 | packages/quant_pod/mcp/server.py | Added _read_memory_file(); injects strategy_context + session_notes into every run_analysis crew run; wired ICOutputValidator into _populate_ic_cache_from_result | Crew had no awareness of active strategies or cross-session findings | Determinism audit |
| 2026-03-15 | packages/quant_pod/prompts/assistant/trading_assistant.json | Rewrote to require JSON output; disabled reasoning; added explicit JSON field constraints | Assistant was producing ═══ prose; DailyBrief always fell through to raw_output | Determinism audit |
| 2026-03-15 | packages/quant_pod/crews/config/tasks.yaml | assistant_synthesis_task: prose expected_output → strict JSON schema; added {strategy_context}/{session_notes} template vars | Same root cause as assistant JSON fix | Determinism audit |
| 2026-03-15 | packages/quant_pod/guardrails/ic_output_validator.py | New: ICOutputValidator with per-IC required field patterns; wired into _populate_ic_cache_from_result | No IC output validation existed; /tune had no evidence base | Determinism audit |
| 2026-03-15 | .claude/skills/tune.md | New skill: /tune for IC+pod prompt improvement; wired into /reflect Step 4.5 | No process existed to improve IC prompts from reflect findings | Determinism audit |
| 2026-03-16 | packages/quantcore/data/_ohlcv.py | Fixed save_ohlcv: drop non-core columns (vwap, trade_count) before 6-column rename | Alpaca returns 8 cols, save_ohlcv hardcoded 6; silent write failure prevented data persistence | XOM/MSFT daily data never reached DuckDB |
| 2026-03-16 | packages/quant_pod/mcp/tools/strategy.py:289 | Fixed update_strategy: `get_strategy()` → `get_strategy.fn()` | FunctionTool objects aren't directly callable; needs `.fn()` for internal calls | `update_strategy` returned "'FunctionTool' object is not callable" |
