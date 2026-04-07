<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-docker-security
section-02-tool-ordering
section-03-prompt-caching
section-04-hnsw-index
section-05-rag-fix
section-06-sentiment-fallback
section-07-validation-and-wiring
END_MANIFEST -->

# Implementation Sections Index

## Dependency Graph

| Section | Depends On | Blocks | Parallelizable |
|---------|------------|--------|----------------|
| section-01-docker-security | - | - | Yes |
| section-02-tool-ordering | - | section-03 | Yes |
| section-03-prompt-caching | section-02 | - | No |
| section-04-hnsw-index | - | section-05 | Yes |
| section-05-rag-fix | section-04 | - | No |
| section-06-sentiment-fallback | - | - | Yes |
| section-07-validation-and-wiring | 01-06 | - | No |

## Execution Order

1. **Batch 1** (parallel, no dependencies): section-01-docker-security, section-02-tool-ordering, section-04-hnsw-index, section-06-sentiment-fallback
2. **Batch 2** (parallel, after batch 1): section-03-prompt-caching (needs 02), section-05-rag-fix (needs 04)
3. **Batch 3** (sequential, after all): section-07-validation-and-wiring

## Section Summaries

### section-01-docker-security
Items 0.6 + 0.7: Bind all Docker ports to 127.0.0.1, remove default password fallbacks from docker-compose.yml, add password validation to start.sh, update .env.example. Tests: start.sh validation with bad env vars.

### section-02-tool-ordering
Item 0.1: Sort tools by name in get_tools_for_agent() and get_tools_for_agent_with_search(), keep TOOL_SEARCH_TOOL last. Tests: sorting stability, TOOL_SEARCH_TOOL position.

### section-03-prompt-caching
Item 0.2: Add cache_control breakpoints on tools and system messages (provider-aware), add cache hit rate logging to observability. Tests: provider-aware SystemMessage construction, tool cache_control placement, non-Anthropic negative tests.

### section-04-hnsw-index
Item 0.4: Add HNSW vector index migration to db.py's run_migrations(). Tests: migration creates index, idempotency, correct operator class and parameters.

### section-05-rag-fix
Item 0.3: Replace SQL in search_knowledge_base tool with call to rag.query.search_knowledge_base(), map return schema, handle Ollama dependency. Includes pre-implementation data investigation. Tests: RAG integration, schema mapping, Ollama failure handling.

### section-06-sentiment-fallback
Item 0.5: Audit active sentiment collector (sentiment.py vs sentiment_alphavantage.py), change _safe_defaults() to return {}, verify synthesis handles empty dict. Tests: fallback returns {}, synthesis handles {}.

### section-07-validation-and-wiring
Cross-cutting validation: verify all changes work together, run full test suite, check for regressions. Manual validation per the validation plan (Langfuse cost comparison, semantic search verification, Docker security check).
