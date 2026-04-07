# Phase 5: Cost Optimization — Implementation Plan

---

## 1. Background and Motivation

QuantStack is an autonomous trading system running 21 LLM-powered agents across 3 LangGraph StateGraphs (Research, Trading, Supervisor) on 5-10 minute cycles. At current rates, system prompt tokens alone cost ~$126/day. A CTO audit identified 8 cost optimization items that together reduce LLM spend by 60-80%, saving $40K-$64K annually (combined with Phase 0 caching work already complete).

The system runs as Docker services orchestrated by `start.sh`: PostgreSQL + pgvector, Langfuse (observability), Ollama (local models), and 3 graph services.

**Critical: Dual LLM Config Systems.** The codebase has two independent LLM routing layers:
1. `src/quantstack/llm/provider.py` + `src/quantstack/llm/config.py` — tier system (heavy/medium/light), used by graph agent executor
2. `src/quantstack/llm_config.py` — legacy tier system (IC/Pod/Assistant/Decoder/Workshop), used by some standalone scripts and tools (hypothesis_agent.py, opro_loop.py, sentiment collectors)

These must be consolidated before LiteLLM deployment. This plan adds a prerequisite item 5.0 for this consolidation.

This plan covers items 5.0-5.8 implemented in spec order. Each item is designed to be independently deployable — earlier items don't block later ones except where noted.

---

## 1A. Prerequisite Item 5.0: Consolidate Dual LLM Config Systems

### 1A.1 Problem

Two independent LLM routing systems coexist: `llm/provider.py` (heavy/medium/light tiers) and `llm_config.py` (IC/Pod/Assistant/Decoder/Workshop tiers). Each has its own provider fallback chain, env var overrides (`LLM_MODEL_IC`, `LLM_MODEL_POD`, etc.), and model string resolution. This means items 5.3 (tier reclassification), 5.6 (hardcoded strings), and 5.7 (LiteLLM) would each need to address both systems independently — doubling the work and risk.

### 1A.2 Approach

1. **Audit usage:** Grep for `from quantstack.llm_config import` and `from quantstack.llm.config import` to identify all consumers of each system.
2. **Map tiers:** IC → light, Pod → heavy, Assistant → heavy, Decoder → light, Workshop → heavy. Validate these mappings against actual model assignments in each config.
3. **Migrate callers:** Update all `llm_config.py` consumers to use `get_chat_model(tier)` from `llm/provider.py`. For env var overrides (`LLM_MODEL_IC`, etc.), add backward-compatible aliases that map to the canonical tier system.
4. **Deprecate and remove:** Mark `llm_config.py` as deprecated, then remove after all callers are migrated.

### 1A.3 Risk

Low risk — this is a rename/redirect, not a behavior change. Each caller should produce identical LLM calls after migration. Verify via Langfuse traces that model IDs match before and after.

---

## 2. Item 5.1: Context Compaction at Merge Points

### 2.1 Problem

The Trading graph has two merge points (`merge_parallel` and `merge_pre_execution`) that are currently no-op convergence nodes. All upstream agent outputs — position reviews, exit signals, entry scans, earnings analysis, portfolio construction, options analysis — flow through unmodified. By the time `execute_entries` runs, it receives 65-120KB of accumulated context per cycle. The only guard is reactive message pruning at 150K chars in `agent_executor.py`, which drops the oldest tool rounds and can lose critical early-cycle data.

### 2.2 Architecture

Replace each no-op merge with a **deterministic compaction node** that extracts structured fields from typed graph state into a Pydantic-typed brief. No LLM call required — the Trading graph state is already typed with well-defined keys, so Python code can extract and reshape the fields directly. This is faster, cheaper, and more reliable than LLM-based summarization.

**Why deterministic over LLM-based:** The upstream agent outputs are structured data (exit signals, entry candidates, risk verdicts) stored in typed state keys. A Python function that reads these keys and populates a Pydantic model achieves the same 40%+ context reduction without adding latency, cost, or a failure point to the critical trading path. LLM-based compaction would only be needed for unstructured prose fields — evaluate adding it as a v2 enhancement only if deterministic compaction doesn't meet the reduction target.

**Two compaction nodes:**

1. **`compact_parallel`** — replaces `merge_parallel`, runs after `execute_exits` and `entry_scan`/`earnings_analysis` converge. Produces a `ParallelMergeBrief`.

2. **`compact_pre_execution`** — replaces `merge_pre_execution`, runs after `portfolio_review` and `analyze_options` converge. Produces a `PreExecutionBrief`.

### 2.3 Brief Schemas

Define in a new file `src/quantstack/graphs/trading/briefs.py` using **Pydantic BaseModel** (required for compatibility with LangChain's `with_structured_output()` if LLM-based v2 is added later):

```python
class ExitAction(BaseModel):
    symbol: str
    action: str          # HOLD, TRIM, CLOSE
    reason: str

class EntryCandidate(BaseModel):
    symbol: str
    signal_strength: float = Field(ge=0.0, le=1.0)
    thesis: str
    ewf_bias: str | None = None

class RiskFlag(BaseModel):
    risk_type: str       # correlation, sector_concentration, drawdown, etc.
    severity: str        # low, medium, high
    detail: str

class ParallelMergeBrief(BaseModel):
    exits: list[ExitAction]
    entries: list[EntryCandidate]
    risks: list[RiskFlag]
    regime: str
    earnings_flags: list[str]

class ApprovedEntry(BaseModel):
    symbol: str
    position_size: float
    structure: str       # equity, options_spread, etc.
    rationale: str

class OptionsSpec(BaseModel):
    symbol: str
    legs: list[dict]
    max_loss: float
    target_profit: float

class PreExecutionBrief(BaseModel):
    approved: list[ApprovedEntry]
    rejected: list[dict]   # {symbol, reason}
    options_specs: list[OptionsSpec]
    risk_checks: dict      # correlation_ok, capital_ok, sector_ok
```

### 2.4 Compaction Node Implementation

Each compaction node is a **deterministic Python function** (no LLM call):

1. Read all relevant typed state keys from the graph state
2. Extract and reshape fields into the brief Pydantic model
3. Validate via Pydantic (catches missing/malformed data)
4. Write the brief to graph state under a dedicated key (`parallel_brief`, `pre_execution_brief`)
5. Downstream nodes read the brief key instead of raw upstream keys

**Fallback on validation failure:** If Pydantic validation fails (unexpected state shape), log a warning and pass a degraded brief with whatever fields populated successfully, plus a `compaction_degraded: true` flag. Never block the trading cycle on a compaction failure.

### 2.4A Downstream Agent Brief Consumption

Agents downstream of each merge point need their system prompts and state injection updated:

- **`risk_sizing`** (deterministic node after `compact_parallel`): Currently reads raw entry candidates and exit signals. Update to read from `parallel_brief.entries` and `parallel_brief.exits`.
- **`execute_entries`** (agent after `compact_pre_execution`): Currently reads from multiple state keys. Update to read from `pre_execution_brief.approved` and `pre_execution_brief.options_specs`. The agent's system prompt should reference brief fields.
- **`reflect`** (agent after `execute_entries`): Needs both the brief AND execution outcomes. Inject `pre_execution_brief` as context alongside execution results.

In `agent_executor.py`, when building the system message for an agent, serialize the relevant brief as a structured section. The brief's Pydantic `.model_dump_json()` produces clean, parseable JSON that agents can reference.

### 2.5 Graph Wiring Changes

In `src/quantstack/graphs/trading/graph.py`:

- Replace the `merge_parallel` edge target with `compact_parallel` node
- Replace the `merge_pre_execution` edge target with `compact_pre_execution` node
- Update downstream nodes (`risk_sizing`, `execute_entries`) to read from brief state keys
- Keep raw state keys available in checkpoint for debugging/audit (don't delete them)

### 2.6 Measuring Impact

Before deploying, instrument context size measurement:
- Log `len(str(state))` at `execute_entries` entry point (before and after compaction)
- Log token count of the brief vs. raw state
- Target: 40%+ reduction in context tokens at `execute_entries`

### 2.7 Risks and Mitigations

- **Data loss in compaction:** Brief schema must cover all fields that downstream agents actually use. Audit `execute_entries` and `reflect` agent prompts to confirm. Keep raw data in checkpoint (not deleted, just not passed to LLM).
- **Haiku extraction failures:** If structured output parsing fails, fall back to passing raw state (degrade gracefully, don't block the cycle).
- **Brief staleness:** The brief is a snapshot. If a downstream agent needs a field not in the brief, add it to the schema — don't bypass compaction.

---

## 3. Item 5.2: Per-Agent Cost Tracking via Langfuse

### 3.1 Problem

No per-agent cost aggregation exists. Langfuse captures raw token counts per LLM call, but there's no way to answer "which agent costs the most per day?" or enforce token budgets.

### 3.2 Architecture

Use Langfuse as the source of truth. Enrich LLM call metadata with agent/graph identifiers, then build aggregation queries on top.

### 3.3 Metadata Enrichment

In `src/quantstack/graphs/agent_executor.py`, ensure every LLM invocation includes these Langfuse metadata tags:

- `agent_name` — from `AgentConfig.name`
- `graph_name` — from graph context (research/trading/supervisor)
- `cycle_id` — unique ID per graph invocation cycle
- `llm_tier` — from `AgentConfig.llm_tier`

The existing `langfuse_trace_context()` in `src/quantstack/observability/tracing.py` creates traces per graph cycle. Within each trace, individual LLM calls should be logged as "generations" with the above metadata. Verify the current `log_llm_call()` function in `instrumentation.py` passes these fields; add them if missing.

### 3.4 Aggregation Queries

Build a set of Langfuse API queries (or use Langfuse's dashboard/SQL access if available) for:

1. **Per-agent daily cost:** Sum token costs grouped by `agent_name` and date
2. **Per-graph cycle cost:** Sum all generation costs within a trace
3. **Top-N expensive agents:** Ranked by 7-day rolling cost
4. **Cost trend:** Daily cost over time, by agent and graph
5. **Anomaly detection:** Flag any agent whose cycle cost exceeds 3x its 7-day average

Package these as utility functions in a new `src/quantstack/observability/cost_queries.py` module. The Supervisor graph's `health_monitor` agent can call these to detect cost anomalies.

### 3.5 Token Budget Enforcement

Add `max_tokens_budget: int | None` to `AgentConfig` dataclass in `src/quantstack/graphs/config.py`. Default `None` (no limit). Configure per agent in `agents.yaml`.

Enforcement point: the tool-calling loop in `agent_executor.py`. After each LLM response, accumulate `prompt_tokens + completion_tokens`. If cumulative exceeds `max_tokens_budget`:

1. Set a `budget_exceeded` flag in the agent's output
2. Return the last LLM response as-is (partial result)
3. Do NOT make another tool call or LLM invocation
4. Log the budget breach as a Langfuse event

This is a graceful exit — the agent produces whatever it has so far. Downstream nodes should check for `budget_exceeded` and handle accordingly (e.g., `execute_entries` skips candidates that lack full analysis).

### 3.6 Alerting

Add a cost check to the Supervisor graph's `health_monitor` scheduled task:
- Query Langfuse for current-day agent costs
- If any agent exceeds its daily cost threshold (configurable in `agents.yaml`), emit an alert via the existing alerting mechanism
- If total system cost exceeds a global daily cap, trigger the kill switch

---

## 4. Item 5.3: Agent Tier Reclassification

### 4.1 Problem

Despite explicit `llm_tier` fields in `agents.yaml`, some agents still land on incorrect models due to naming-convention-based fallback logic in the LLM routing layer.

### 4.2 Approach

1. **Audit current state:** Query Langfuse traces for the last 7 days. For each agent, extract the actual model ID used. Compare against the intended tier in `agents.yaml`. Document any mismatches.

2. **Remove default-tier fallback:** In `src/quantstack/llm/provider.py`, find the naming-convention-based classification logic. Replace it with an explicit lookup that logs a WARNING if an agent name doesn't match any configured tier. The warning should include the agent name and the tier it would have defaulted to. Never silently assign a default — make it loud.

3. **Review tier assignments:** Based on each agent's actual computational needs:
   - **Heavy tier** (Sonnet-class, $3/MTok): Agents that do complex reasoning, multi-step analysis, or generate novel strategies. Keep: `quant_researcher`, `ml_scientist`, `strategy_rd`, `trade_debater`, `fund_manager`, `options_analyst`, `domain_researcher`, `execution_researcher`.
   - **Medium tier** (Haiku-class, $0.25/MTok): Agents that do structured extraction, monitoring, or template-based analysis. Keep: `hypothesis_critic`, `community_intel`, `daily_planner`, `position_monitor`, `exit_evaluator`, `earnings_analyst`, `market_intel`, `trade_reflector`, `executor`, `self_healer`, `portfolio_risk_monitor`, `strategy_promoter`.
   - **Light tier** (Haiku-class, cheapest): Simple coordination, health checks. Keep: `health_monitor`.

4. **Measure impact:** Compare 7-day rolling cost before and after reclassification using the cost queries from item 5.2.

### 4.3 Key Change

The critical code change is in `get_chat_model()` or its helper that resolves agent name → tier. It must:
- Accept an explicit tier parameter (already does)
- Never fall back to a default tier based on name patterns
- Log a warning if the tier parameter is missing or unrecognized

---

## 5. Item 5.4: Per-Agent Temperature Config

### 5.1 Problem

All agents use `temperature=0.0` globally. Research agents that need diversity in hypothesis generation are artificially constrained; execution agents that need determinism are already correct.

### 5.2 Schema Change

Add `temperature: float | None = None` to `AgentConfig` in `src/quantstack/graphs/config.py`. When `None`, inherit the global default (0.0).

Add `temperature:` field to every agent in all three `agents.yaml` files.

### 5.3 Wiring

**Signature change to `get_chat_model()`:**

```python
def get_chat_model(
    tier: str,
    thinking: dict | None = None,
    temperature: float | None = None,   # NEW — None means use ModelConfig default (0.0)
) -> BaseChatModel:
    """Return a configured LangChain ChatModel for the given tier."""
```

`temperature=None` preserves backward compatibility — the ~30+ existing call sites that pass only `tier` (and optionally `thinking`) continue to work unchanged, getting the default 0.0.

In `agent_executor.py`, when instantiating the LLM for an agent, pass `config.temperature` through:

```python
def get_agent_llm(config: AgentConfig) -> BaseChatModel:
    """Get LLM for agent, respecting per-agent temperature."""
    # config.temperature (from agents.yaml) → get_chat_model(temperature=...)
```

After LiteLLM integration (5.7), temperature becomes a parameter on the LiteLLM request body.

### 5.4 Recommended Values

| Temperature | Agents | Rationale |
|-------------|--------|-----------|
| 0.7 | quant_researcher, hypothesis_critic | Maximum ideation diversity. hypothesis_critic at medium tier + 0.7 temp is intentional: it scores hypotheses on a 0-1 scale and benefits from diverse scoring perspectives without needing heavy-tier reasoning. |
| 0.3-0.5 | trade_debater, community_intel | Moderate diversity for debate/exploration |
| 0.1 | daily_planner, position_monitor, market_intel | Slight variation, mostly deterministic |
| 0.0 | executor, fund_manager, exit_evaluator, risk nodes, all supervisor agents | Determinism required for execution and safety |

---

## 6. Item 5.5: EWF Deduplication via Graph State

### 6.1 Problem

5 agents independently call `get_ewf_analysis` for the same symbols in the same cycle, producing 3-5x redundant database queries.

### 6.2 Architecture

Add an `ewf_cache` dictionary to the Trading graph state. Pre-populate it early in the cycle; all agents read from it instead of querying the DB directly.

### 6.3 State Schema Change

In the Trading graph's state definition, add:

```python
ewf_cache: dict[str, dict]  # key: "{symbol}:{timeframe}", value: EWF analysis result
```

### 6.4 Cache Population

In the `data_refresh` node (first deterministic node in the Trading graph), after refreshing market data:

1. Get the current watchlist symbols from state
2. For each symbol, fetch EWF analysis for all relevant timeframes (4h, daily)
3. Also fetch blue box setups for the current date
4. Store all results in `ewf_cache`

### 6.5 Tool Modification — Cache Injection Mechanism

**Approach: Module-level cache dict.** The `data_refresh` node writes to a module-level `_ewf_cycle_cache` dict in `ewf_tools.py`. The tool functions check this cache before querying the DB. This is the simplest approach for this codebase — no changes to the tool registry, agent executor, or tool signatures needed.

```python
# Module-level in ewf_tools.py
_ewf_cycle_cache: dict[str, dict] = {}

def populate_ewf_cache(symbols: list[str], timeframes: list[str]) -> None:
    """Called by data_refresh node at cycle start. Populates module cache."""

def clear_ewf_cache() -> None:
    """Called at cycle end or on error. Clears stale cache."""
```

Modify `get_ewf_analysis`:
1. Check `_ewf_cycle_cache` first: if `{symbol}:{timeframe}` key exists, return cached result
2. Fall back to DB query only on cache miss (handles symbols not in the pre-fetch list)
3. Same pattern for `get_ewf_blue_box_setups`

The tool's external API (parameters, return type) stays the same. The cache is an internal optimization invisible to the agents.

**Clearing:** Call `clear_ewf_cache()` at cycle end (in `reflect` node or via graph cleanup). This ensures no cross-cycle staleness.

### 6.6 Research Graph

The Research graph also has 2 agents calling `get_ewf_analysis` (quant_researcher, domain_researcher). Apply the same pattern: add `ewf_cache` to Research graph state, populate in `context_load` node.

---

## 7. Item 5.6: Remove Hardcoded Model Strings

### 7.1 Problem

Multiple files hardcode provider-specific model strings (e.g., `anthropic/claude-sonnet-4-20250514`, `gpt-4o-mini`, `groq/llama-3.3-70b-versatile`) instead of using `get_chat_model()`.

### 7.2 Approach

1. **Comprehensive grep:** Search `src/quantstack/` for all model string patterns outside of `llm/config.py`:
   - `claude-sonnet`, `claude-haiku`, `claude-opus`
   - `gpt-4o`, `gpt-4`, `gpt-3.5`
   - `llama-3`, `llama3`
   - `anthropic/`, `bedrock/`, `openai/`, `groq/`, `gemini/`, `ollama/`

2. **For each hardcoded string:**
   - Determine what tier it represents (heavy, medium, light, embedding)
   - Replace with `get_chat_model(tier)` call
   - If the model serves a specific purpose not covered by existing tiers (e.g., bulk processing, embedding), add a named tier to `config.py`

3. **Known locations from audit** (verify these still exist):
   - `tool_search_compat.py:21` — Sonnet hardcoded for tool search compatibility
   - `trading/nodes.py:843` — Sonnet hardcoded for a specific node
   - `trade_evaluator.py` — Sonnet hardcoded
   - `mem0_client.py` — `gpt-4o-mini` hardcoded (for Mem0 integration)
   - `hypothesis_agent.py` — Groq/Llama hardcoded
   - `opro_loop.py` — Groq/Llama hardcoded

4. **Special cases:**
   - `mem0_client.py` uses `gpt-4o-mini` because Mem0 may require OpenAI-compatible API. After LiteLLM (5.7), route through LiteLLM's OpenAI-compatible endpoint instead.
   - OPRO/TextGrad loops may intentionally use specific providers for cost. Add `bulk` tier for these use cases.

### 7.3 Validation

After all replacements: `grep -rn "claude-sonnet\|gpt-4o\|llama-3\|anthropic/\|bedrock/\|openai/\|groq/\|gemini/" src/quantstack/ --include="*.py"` should return only hits in `llm/config.py`.

### 7.4 Regression Test

Add a pytest test (`tests/unit/test_no_hardcoded_models.py`) that greps `src/quantstack/` for hardcoded model strings, excluding `llm/config.py` and `litellm_config.yaml`. The test fails if any matches are found. This prevents regressions — any future contributor who hardcodes a model string will break CI.

---

## 8. Item 5.7: LiteLLM Proxy Deployment

### 8.1 Problem

Provider availability is checked at startup only. A mid-execution 429 or API credit exhaustion (as happened 2026-04-05) blocks the entire system with no automatic recovery.

### 8.2 Architecture

Deploy LiteLLM as a Docker service. All LLM calls route through it. LiteLLM handles provider fallback, circuit breaking, retries, and cost tracking. `get_chat_model()` becomes a thin wrapper that returns a LangChain ChatModel pointed at the LiteLLM proxy.

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  Graph Services  │────▶│   LiteLLM    │────▶│  Anthropic   │
│  (Research,      │     │   Proxy      │────▶│  Bedrock     │
│   Trading,       │     │  :4000       │────▶│  Groq        │
│   Supervisor)    │     │              │────▶│  Ollama      │
└─────────────────┘     └──────────────┘     └─────────────┘
```

### 8.3 Docker Compose Addition

Add a `litellm` service to `docker-compose.yml`:

- **Image:** `ghcr.io/berriai/litellm:main-latest` (or pin a specific version)
- **Port:** 4000 (internal Docker network only, not exposed to host unless needed for debugging)
- **Volumes:**
  - Mount `litellm_config.yaml` (model definitions, routing rules)
  - Mount Zscaler cert bundle (`~/.zscaler_certifi_bundle.pem`) to `/etc/ssl/certs/zscaler.pem`
- **Environment:**
  - `SSL_CERT_FILE=/etc/ssl/certs/zscaler.pem`
  - `REQUESTS_CA_BUNDLE=/etc/ssl/certs/zscaler.pem`
  - `ANTHROPIC_API_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME` (for Bedrock)
  - `GROQ_API_KEY`
  - `LITELLM_MASTER_KEY` (for admin API access)
- **Health check:** `curl -f http://localhost:4000/health`
- **Depends on:** nothing (standalone service)

### 8.4 LiteLLM Configuration

Create `litellm_config.yaml` with:

**Model definitions** — use the same tier names as the existing system (`heavy`, `medium`, `light`) to avoid a translation layer:

- `heavy`: Bedrock Sonnet (order=1) → Anthropic Sonnet (order=2) → Groq Llama-70b (order=3, degraded capability)
- `medium`: Bedrock Haiku (order=1) → Anthropic Haiku (order=2)
- `light`: Bedrock Haiku (order=1) → Anthropic Haiku (order=2)
- `embedding`: Ollama mxbai-embed-large (order=1, local)
- `bulk`: Groq Llama-70b (order=1, cheapest) → Bedrock Haiku (order=2) — for OPRO/TextGrad loops

**Router settings:**
- `allowed_fails: 3` — failures before cooldown
- `cooldown_time: 60` — seconds deployment is unavailable after cooldown
- `num_retries: 2` — retries per request before fallback
- `retry_after: 5` — minimum seconds between retries
- Retry policy: `RateLimitErrorRetries=3, TimeoutErrorRetries=2, AuthenticationErrorRetries=0`
- `enable_pre_call_checks: true` — validate context window before sending

**Context window fallbacks:**
- `quantstack-heavy` → `quantstack-heavy-large-context` (for research agents that accumulate long contexts)

### 8.5 Refactoring get_chat_model()

Transform `src/quantstack/llm/provider.py`:

**Before:** `get_chat_model(tier)` instantiates provider-specific ChatModel classes directly (ChatBedrock, ChatAnthropic, etc.) with fallback logic.

**After:** `get_chat_model(tier)` returns a `ChatOpenAI` instance (LangChain's OpenAI-compatible client) pointed at the LiteLLM proxy URL (`http://litellm:4000/v1`). The model name is the tier name directly (`heavy`, `medium`, `light`) — no translation needed since LiteLLM uses the same names.

The existing `ProviderConfig`, `FALLBACK_ORDER`, and provider-specific initialization logic can be removed or kept as fallback for running without LiteLLM (useful for local development).

Add an env var `LITELLM_PROXY_URL` (default: `http://litellm:4000`). When set, all routing goes through LiteLLM. When unset, fall back to direct provider routing (backward compatibility).

### 8.6 Zscaler Considerations

LiteLLM makes outbound HTTPS calls to Anthropic, AWS (Bedrock), and Groq APIs. Behind Zscaler, these calls will fail with `certificate verify failed` unless the Zscaler cert bundle is trusted.

The Docker Compose mount handles this: map the host's `~/.zscaler_certifi_bundle.pem` into the container and set `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` environment variables. Verify connectivity to all providers during the health check.

### 8.7 Operational Events

Configure LiteLLM callbacks to log provider switches and failures to Langfuse:
- On fallback: log which provider failed, which took over, and the error
- On cooldown entry/exit: log the deployment and duration
- On budget breach: log the agent and amount

### 8.8 Migration Strategy

1. Deploy LiteLLM in Docker Compose (additive, no existing services changed)
2. Test connectivity to all providers through LiteLLM
3. Set `LITELLM_PROXY_URL` in one graph service (Supervisor — lowest risk)
4. Verify traces in Langfuse show LiteLLM routing
5. Roll out to Trading graph, then Research graph
6. Remove direct provider initialization code after all graphs confirmed working

---

## 9. Item 5.8: Memory Temporal Decay

### 9.1 Problem

Memory entries in both PostgreSQL (`agent_memory` table) and `.claude/memory/` markdown files accumulate indefinitely. Stale entries waste tokens when injected into agent context. The 2026-04-04 EWF market reads flagged as "low-trust" still persist and consume context.

### 9.2 PostgreSQL Memory Lifecycle

**Schema changes** to `agent_memory` table (via migration in `src/quantstack/db/`):

Add two columns:
- `last_accessed_at TIMESTAMP DEFAULT NOW()` — updated on every read
- `archived_at TIMESTAMP DEFAULT NULL` — set when entry is archived

Create archive table:
- `agent_memory_archive` — identical schema to `agent_memory`, receives archived rows

**Read-path change** in `src/quantstack/memory/blackboard.py`:
- On every `read_recent()` or `read_as_context()` call, update `last_accessed_at` for returned rows
- Apply temporal decay weighting in SQL for efficiency:

```sql
SELECT *, POW(0.5, EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0 / :half_life) AS decay_weight
FROM agent_memory
WHERE symbol = :symbol AND archived_at IS NULL
ORDER BY decay_weight DESC
LIMIT :limit
```

This avoids the over-fetch problem (fetching all rows then sorting in Python). PostgreSQL computes the decay weight per row and returns only the top-N by weighted relevance. The `half_life` parameter is looked up from a category→half_life mapping based on each row's `category` column. For categories with different half-lives in the same query, use a CASE expression in SQL.

**Half-life by category:**

| Category | Half-life (days) | Rationale |
|----------|-----------------|-----------|
| trade_outcome | 14 | Recent trades most relevant |
| strategy_param | 30 | Parameters evolve slowly |
| market_regime | 7 | Regimes shift quickly |
| research_finding | 90 | Foundational knowledge |
| general | 30 | Default |

**Weekly pruning job** — add to Supervisor graph's `scheduled_tasks` node:
1. Query `agent_memory` for rows where `created_at < NOW() - (half_life * 3)` (effectively <12.5% relevance)
2. `INSERT INTO agent_memory_archive SELECT * FROM agent_memory WHERE ...`
3. `DELETE FROM agent_memory WHERE ...`
4. Log pruning stats: rows archived, by category, total freed

### 9.3 Markdown Memory Lifecycle

**Date metadata:** Ensure all `.claude/memory/` files have `created:` in their YAML frontmatter. For existing files without dates, infer from git blame or file modification time.

**TTL enforcement** — enhance the `compact-memory` skill:

| Memory type | TTL | Action |
|-------------|-----|--------|
| Market reads (EWF, regime) | 7 days | Archive to `*.archive.md` |
| Session handoffs | 30 days | Archive |
| Strategy states | Monthly validation | Flag for review, don't auto-archive |
| Workshop lessons | Permanent | Never archive |
| Validated principles | Permanent | Never archive |

**Archival format:** Move expired entries to `<filename>.archive.md` in the same directory. Prefix each archived entry with `[ARCHIVED <date>]`. This is recoverable — just move entries back.

### 9.4 Integration with Compaction

After 5.1 (context compaction) is deployed, the temporal decay weighting in `read_as_context()` means that stale memories are naturally de-emphasized even before they hit their TTL. This provides a smooth degradation curve rather than a hard cutoff.

---

## 10. Cross-Cutting Concerns

### 10.1 AgentConfig Schema Evolution

Items 5.2 and 5.4 both add fields to `AgentConfig`:
- `max_tokens_budget: int | None = None` (5.2)
- `temperature: float | None = None` (5.4)

Implement both in a single schema change to avoid churn. Update all three `agents.yaml` files with the new fields.

### 10.2 Testing Strategy

Each item needs tests. The existing test infrastructure uses pytest with markers. Key test files to extend:

- `tests/unit/test_llm_provider.py` — add tests for LiteLLM routing, fallback behavior, tier resolution without defaults
- `tests/unit/test_agent_executor.py` — add tests for budget enforcement, temperature passing
- `tests/unit/test_agent_configs.py` — add validation for new fields (temperature, budget), ensure no agent missing explicit tier
- New: `tests/unit/test_compaction.py` — brief schema validation, compaction node logic
- New: `tests/unit/test_ewf_cache.py` — cache hit/miss behavior, graph state integration
- New: `tests/unit/test_memory_decay.py` — temporal decay weighting, archival logic
- New: `tests/integration/test_litellm.py` — provider fallback simulation, circuit breaker

### 10.3 Rollback Plan

Each item is independently deployable and reversible:
- **5.1:** Remove compaction nodes, restore no-op merges
- **5.2:** Remove budget enforcement from agent_executor (cost tracking is additive, no rollback needed)
- **5.3:** Revert agents.yaml tier changes
- **5.4:** Set all temperatures back to null (defaults to 0.0)
- **5.5:** Remove ewf_cache from graph state, tools revert to direct DB queries
- **5.6:** Not reversible in practice (hardcoded strings shouldn't come back), but get_chat_model() is the stable API
- **5.7:** Unset `LITELLM_PROXY_URL` to fall back to direct provider routing
- **5.8:** Drop archive table columns, disable pruning job

### 10.4 Observability

All items should produce measurable signals in Langfuse:
- **5.1:** Context token count before/after compaction (log as custom event)
- **5.2:** Per-agent cost is the primary new metric
- **5.3:** Model ID per agent per trace (already captured)
- **5.4:** Temperature per agent per trace
- **5.5:** EWF query count per cycle (log cache hits vs misses)
- **5.6:** Zero hardcoded model strings (grep-based CI check)
- **5.7:** Provider switch events, cooldown events, fallback counts
- **5.8:** Memory store size, archival counts, average memory age

---

## 11. File Change Summary

| File | Changes | Items |
|------|---------|-------|
| `src/quantstack/llm_config.py` | Deprecate and remove — migrate callers to llm/provider.py | 5.0 |
| `src/quantstack/graphs/trading/graph.py` | Replace merge nodes with compaction nodes | 5.1 |
| `src/quantstack/graphs/trading/briefs.py` (new) | Pydantic brief schemas | 5.1 |
| `src/quantstack/graphs/trading/compaction.py` (new) | Compaction node implementations | 5.1 |
| `src/quantstack/graphs/config.py` | Add temperature, max_tokens_budget to AgentConfig | 5.2, 5.4 |
| `src/quantstack/graphs/*/config/agents.yaml` (×3) | New fields, tier corrections | 5.2, 5.3, 5.4 |
| `src/quantstack/graphs/agent_executor.py` | Budget enforcement, temperature wiring | 5.2, 5.4 |
| `src/quantstack/observability/cost_queries.py` (new) | Langfuse aggregation queries | 5.2 |
| `src/quantstack/observability/instrumentation.py` | Enrich LLM call metadata with agent/graph tags | 5.2 |
| `src/quantstack/llm/provider.py` | Remove default-tier fallback, add LiteLLM routing | 5.3, 5.7 |
| `src/quantstack/llm/config.py` | Add bulk tier, remove naming conventions | 5.3, 5.6 |
| `src/quantstack/tools/langchain/ewf_tools.py` | Add graph-state cache lookup | 5.5 |
| `src/quantstack/graphs/trading/state.py` or equivalent | Add ewf_cache to graph state | 5.5 |
| `src/quantstack/graphs/trading/nodes.py` | Pre-fetch EWF in data_refresh | 5.5 |
| Various files with hardcoded strings | Replace with get_chat_model() | 5.6 |
| `docker-compose.yml` | Add litellm service | 5.7 |
| `litellm_config.yaml` (new) | LiteLLM model definitions, routing rules | 5.7 |
| `src/quantstack/memory/blackboard.py` | Temporal decay, last_accessed_at tracking | 5.8 |
| `src/quantstack/db/migrations/` (new) | agent_memory schema changes, archive table | 5.8 |
| `.claude/memory/` files | Add date metadata | 5.8 |
| `tests/unit/test_compaction.py` (new) | Brief schema, compaction tests | 5.1 |
| `tests/unit/test_ewf_cache.py` (new) | Cache hit/miss tests | 5.5 |
| `tests/unit/test_memory_decay.py` (new) | Decay weighting, archival tests | 5.8 |
| `tests/unit/test_no_hardcoded_models.py` (new) | Regression test for model strings | 5.6 |
| `tests/integration/test_litellm.py` (new) | Provider fallback tests | 5.7 |
