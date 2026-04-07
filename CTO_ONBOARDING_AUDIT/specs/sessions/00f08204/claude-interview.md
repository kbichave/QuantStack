# Phase 5 Cost Optimization — Interview Transcript

---

## Round 1

### Q1: Agent tier reclassification status

**Q:** The codebase research shows agent tiers are already configured explicitly in agents.yaml (heavy/medium/light). The spec says '10+ agents fall to default Sonnet' — is this finding from the audit now stale, or are there still agents defaulting incorrectly?

**A:** Still needs work. Some agents are still on wrong tiers despite YAML config.

### Q2: Provider fallback architecture

**Q:** For provider fallback (5.7): the research found LiteLLM proxy as a mature solution vs. enhancing the existing get_chat_model() directly. LiteLLM adds a Docker service but handles routing, budgets, and fallback in one layer. Which approach do you prefer?

**A:** Deploy LiteLLM proxy. Add litellm as a Docker service — unified cost tracking, routing, and fallback.

### Q3: Cost tracking approach

**Q:** For per-agent cost tracking (5.2): should we build a custom llm_costs table in PostgreSQL, or leverage Langfuse's existing token tracking with a query/dashboard layer on top?

**A:** Langfuse + aggregation queries. Use Langfuse as source of truth, build views/queries for agent-level rollup.

---

## Round 2

### Q4: Compaction brief format

**Q:** For context compaction (5.1): should the compacted briefs follow a strict schema (TypedDict with exits[], entries[], risks[]) or freeform markdown?

**A:** Strict typed schema. Define Pydantic/TypedDict models for each brief — structured, parseable, testable.

### Q5: Memory TTL scope

**Q:** For memory temporal decay (5.8): which memory system(s) should get TTL treatment? .claude/memory/ files, PostgreSQL agent_memory, or both?

**A:** Both — .claude/memory/ files AND agent_memory table. Full lifecycle management across both storage layers.

### Q6: LiteLLM integration scope

**Q:** Should LiteLLM replace get_chat_model() entirely or sit alongside it?

**A:** Full replacement — all LLM calls go through LiteLLM. get_chat_model() becomes a thin wrapper around LiteLLM client.

---

## Round 3

### Q7: EWF deduplication caching strategy

**Q:** Should the per-cycle EWF cache live in LangGraph graph state or in a short-TTL external cache?

**A:** Graph state — per-invocation only. Fetch once at start of cycle, store in state, all agents read from state. Clean per-cycle.

### Q8: Implementation order

**Q:** What's your preferred implementation order given dependencies and risk?

**A:** As spec ordered (5.1 through 5.8). Follow the spec's item numbering.

### Q9: LiteLLM deployment constraints

**Q:** Can we add LiteLLM to docker-compose.yml alongside existing services? Any Zscaler concerns?

**A:** Docker Compose, but needs Zscaler config. LiteLLM needs to handle Zscaler SSL inspection.

---

## Key Decisions Summary

| Decision | Choice |
|----------|--------|
| Agent tier reclassification | Still needed — some agents on wrong tiers |
| Provider fallback approach | LiteLLM proxy (Docker service) |
| Cost tracking | Langfuse as source of truth + aggregation queries |
| Compaction brief format | Strict typed schema (Pydantic/TypedDict) |
| Memory TTL scope | Both .claude/memory/ files AND agent_memory PG table |
| LiteLLM integration | Full replacement of get_chat_model() routing |
| EWF cache | LangGraph graph state (per-invocation) |
| Implementation order | Spec order (5.1 → 5.8) |
| LiteLLM deployment | Docker Compose with Zscaler cert config |
