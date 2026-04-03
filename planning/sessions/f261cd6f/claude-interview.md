# Interview Transcript: CrewAI to LangGraph Migration

## Q1: Tool Layer Strategy — What to do with 50 mcp_bridge files using crewai_compat.py BaseTool?

**Answer:** Hybrid approach based on who calls the tool:
- **Deterministic tools** (risk calc, data fetch, backtest) → **Plain async functions**. No LLM invokes them; graph nodes call them directly. BaseTool class adds zero value — plain `async def` is cleaner, faster, easier to test.
- **LLM-facing tools** (hypothesis, analysis, signal interpretation) → **LangChain BaseTool**. These need to be LangChain-compatible for LLM discovery/calling through the graph. Coupling to LangChain here is the point.
- **Delete crewai_compat.py entirely** once migration is done.
- Expected split: ~15-20 LangChain tools, ~30-35 plain functions.

## Q2: Agent Config Location (YAML vs Python vs Inline)

**Answer:** YAML → dataclass → graph builder pattern:
- YAML defines agent profiles, rules, backstories (external, non-code iteration on strategy)
- At startup, a loader parses YAML into `@dataclass AgentConfig` with strict validation
- Graph builder takes `AgentConfig` as parameter, stays focused on topology and state transitions
- **Hot-reload is needed** — config changes should take effect without restarting runners

## Q3: Memory/RAG — ChromaDB vs pgvector

**Answer:** **Migrate everything to PostgreSQL + pgvector.** Drop ChromaDB entirely.

## Q4: Migration Strategy — Incremental vs Clean-Cut

**Answer:** **Clean-cut rewrite.** Remove all CrewAI at once, build fresh LangGraph graphs. The spec already maps the full footprint.

## Q5: Tool Wrapper Approach for the 25 crewai_tools Files

**Answer:** **Hybrid — direct for deterministic nodes, @tool for LLM nodes.** Risk calc, backtest, data fetch nodes call MCP directly. Only reasoning nodes (hypothesis, trade analysis) get LLM with @tool.

## Q6: pgvector Migration — Start Fresh or Migrate Existing Embeddings?

**Answer:** **Write a one-time migration script.** Export ChromaDB embeddings and import into pgvector tables.

## Q7: Hot-Reload Mechanism

**Answer:** **Both.** File-watch (watchdog/inotify) in dev, SIGHUP signal-based in production.

## Q8: crewai_docs_md Directory (180 files of crawled CrewAI docs)

**Answer:** **Delete entirely.** No longer needed after migration.

## Q9: mcp_bridge Tool Audit — BaseTool Swap Strategy

**Answer:** Split based on the hybrid decision:
- Audit each file: does an LLM call this, or does a graph node call this?
- LLM caller → LangChain BaseTool
- Node caller → plain async function
- Clean boundary, no zombie abstraction layer

## Q10: Migration Priority Order

**Answer:** No preference — let the plan determine the order based on what makes architectural sense.

## Key Decisions Summary

| Decision | Choice |
|----------|--------|
| Migration strategy | Clean-cut rewrite |
| Tool wrappers | Hybrid: direct MCP for deterministic, @tool for LLM |
| mcp_bridge files | Split: ~15-20 LangChain BaseTool, ~30-35 plain async |
| Agent config | YAML → dataclass → graph builder with hot-reload |
| Hot-reload | File-watch (dev) + SIGHUP (prod) |
| Memory/RAG | pgvector (drop ChromaDB), migration script for embeddings |
| CrewAI docs | Delete entire docs/crewai_docs_md/ |
| crewai_compat.py | Delete after migration |
| Graph order | Architect's choice |
