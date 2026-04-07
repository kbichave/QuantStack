# Integration Notes — Opus Review Feedback

## Integrating

### 1. Section 5: Target file correction (CRITICAL)
**Integrating.** The reviewer is correct that the signal engine may use `collect_sentiment_alphavantage` rather than `collect_sentiment`. The plan must be updated to audit both collectors and fix whichever is actually in the runtime path. This would have caused an implementer to fix dead code while leaving the actual bug untouched.

### 2. Section 2: Provider guard for structured SystemMessage (CRITICAL)
**Integrating.** The fallback chain includes OpenAI and Ollama, which don't support `cache_control` in content blocks. Adding a provider-aware conditional is essential. The plan will be updated to only use structured content blocks when the active provider is Anthropic or Bedrock.

### 3. Section 1: Sort behavior in `get_tools_for_agent_with_search()` (HIGH)
**Integrating.** The reviewer correctly identified that the `tools_for_api` list is built by iterating `tool_names` — sorting the final list isn't enough; `tool_names` must be sorted before the iteration loop. Also integrating the note about `TOOL_SEARCH_TOOL` position.

### 4. Section 3: Return schema mapping (HIGH)
**Integrating.** The field mapping between RAG output and tool contract was left vague. Will specify the exact mapping.

### 5. Section 3: New Ollama dependency (HIGH)
**Integrating.** The RAG function requires Ollama for embedding computation — this is a new runtime dependency the old SQL-only path didn't have. Must document this failure mode and add appropriate error handling.

### 6. Section 3: Connection management mismatch (MEDIUM)
**Integrating.** The plan should specify whether to pass the tool's connection to the RAG function or let RAG manage its own.

### 7. Section 4: Migration location rationale (LOW)
**Integrating.** Will add explicit rationale for why db.py was chosen over rag/query.py.

### 8. Cross-cutting: Single PR contingency for Item 0.3 (MEDIUM)
**Integrating.** If embeddings table is empty, Item 0.3 should be excluded and tracked separately.

### 9. Section 7: Minimum password length (MEDIUM)
**Integrating.** Adding a minimum length check alongside blocklisting known defaults.

### 10. Test gaps: non-Anthropic providers, TOOL_SEARCH_TOOL position, sentiment_alphavantage (HIGH)
**Integrating.** Adding these missing test cases.

## NOT Integrating

### 1. Section 2: Cache creation cost warning
**Not integrating.** The reviewer notes cache creation costs 25% more on first call. This is a transient cost that pays back within the first cache hit. Not worth adding to the plan — it's noise for the implementer.

### 2. Section 2: 5-minute TTL vs agent duration
**Not integrating.** The spec already calls this out as a risk. The plan doesn't need to re-address it.

### 3. Section 6: Container-to-container communication check
**Not integrating.** Docker Compose inter-service communication uses the internal network, not published ports. This is a Docker fundamental — no verification needed.

### 4. Section 7: PR migration instructions
**Not integrating into the plan.** This is a PR description concern, not an implementation concern. The implementer will write appropriate PR description.

### 5. Section 4: ef_search tuning
**Not integrating.** The reviewer's suggestion to set `hnsw.ef_search = 100` is a micro-optimization at this table size. Default 40 is fine.
