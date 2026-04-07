# Phase 0: Quick Wins — TDD Plan

**Testing framework:** pytest (existing project convention)
**Test directory:** `tests/unit/` for unit tests, `tests/integration/` for DB-dependent tests
**Fixtures:** Shared fixtures in `tests/conftest.py`, unit fixtures in `tests/unit/conftest.py`
**Patterns:** `mock_settings` fixture, OHLCV generators, standard mock/patch from `unittest.mock`

---

## Section 1: Deterministic Tool Ordering (Item 0.1)

**Test file:** `tests/unit/test_tool_ordering.py`

```python
# Test: get_tools_for_agent returns tools sorted alphabetically by name
# Setup: Register 3 tools with names "zebra_tool", "alpha_tool", "mid_tool"
# Assert: returned list has names in order ["alpha_tool", "mid_tool", "zebra_tool"]

# Test: get_tools_for_agent_with_search returns sorted deferred and always-loaded lists
# Setup: Register mix of deferred and always-loaded tools with unsorted names
# Assert: both returned lists are sorted by name

# Test: TOOL_SEARCH_TOOL remains last after sorting
# Setup: Register tools where TOOL_SEARCH_TOOL name would sort to middle alphabetically
# Assert: TOOL_SEARCH_TOOL is the last item in tools_for_api list

# Test: sorting is stable across multiple calls
# Setup: Call get_tools_for_agent twice with same tool names
# Assert: identical order both times

# Test: tool_to_anthropic_dict preserves sort order from input
# Setup: Pass pre-sorted tool list
# Assert: output dicts maintain same order as input
```

---

## Section 2: Prompt Caching with Explicit Breakpoints (Item 0.2)

**Test file:** `tests/unit/test_prompt_caching.py`

```python
# Test: build_system_message returns structured content with cache_control for Anthropic provider
# Setup: Call build_system_message with provider="anthropic"
# Assert: returned SystemMessage.content is a list with one dict containing "cache_control"

# Test: build_system_message returns plain string for OpenAI provider
# Setup: Call build_system_message with provider="openai"
# Assert: returned SystemMessage.content is a plain string, no cache_control

# Test: build_system_message returns plain string for Ollama provider
# Setup: Call build_system_message with provider="ollama"
# Assert: returned SystemMessage.content is a plain string

# Test: build_system_message returns structured content for Bedrock provider
# Setup: Call build_system_message with provider="bedrock"
# Assert: returned SystemMessage.content includes cache_control

# Test: last tool in Anthropic dict list has cache_control set
# Setup: Convert 3 sorted tools to Anthropic dicts via the caching path
# Assert: only the last dict has cache_control key, first two do not

# Test: non-Anthropic tool path does not add cache_control
# Setup: Convert tools for a non-Anthropic provider
# Assert: no tool dict has cache_control key
```

---

## Section 3: Fix `search_knowledge_base` to Use RAG (Item 0.3)

**Test file:** `tests/unit/test_knowledge_base_tool.py`

```python
# Test: search_knowledge_base calls rag.query.search_knowledge_base with query
# Setup: Mock rag.query.search_knowledge_base
# Assert: called with the user's query string and top_k

# Test: return schema maps RAG output to tool contract
# Setup: Mock RAG to return [{"text": "foo", "metadata": {}, "distance": 0.1, "collection": "strats"}]
# Assert: tool output contains {"content": "foo", "category": "strats", "metadata": {}, ...}

# Test: empty RAG results return empty list
# Setup: Mock RAG to return []
# Assert: tool returns empty JSON array, no error

# Test: tool handles Ollama unavailability gracefully
# Setup: Mock RAG function to raise ConnectionError (Ollama down)
# Assert: tool returns informative error message, does not crash
```

**Integration test file:** `tests/integration/test_knowledge_base_rag.py`

```python
# Test: semantic search returns relevant results
# Setup: Insert known embeddings for "momentum strategy" and "value investing"
# Assert: query "momentum" returns momentum entry with lower distance than value entry

# Test: query parameter actually affects results (not recency-only)
# Setup: Insert old relevant embedding and new irrelevant embedding
# Assert: old relevant embedding ranks higher than new irrelevant one
```

---

## Section 4: Add HNSW Index on Embeddings (Item 0.4)

**Test file:** `tests/unit/test_hnsw_migration.py`

```python
# Test: migration creates HNSW index on fresh database
# Setup: Run migration on test DB with embeddings table but no HNSW index
# Assert: pg_indexes shows idx_embeddings_hnsw exists

# Test: migration is idempotent
# Setup: Run migration twice on same database
# Assert: no error on second run, index still exists

# Test: index uses correct operator class (vector_cosine_ops)
# Setup: Run migration, query pg_opclass for the index
# Assert: operator class is vector_cosine_ops

# Test: index has correct parameters (m=16, ef_construction=100)
# Setup: Run migration, query pg_index for storage parameters
# Assert: m=16 and ef_construction=100
```

---

## Section 5: Sentiment Fallback Cleanup (Item 0.5)

**Test file:** `tests/unit/test_sentiment_fallback.py`

```python
# Test: active collector's _safe_defaults returns empty dict
# NOTE: First determine which collector is active (sentiment.py or sentiment_alphavantage.py)
# Assert: _safe_defaults() returns {}

# Test: active collector returns {} on timeout
# Setup: Mock the data source client to raise TimeoutError
# Assert: collector returns {}

# Test: active collector returns {} when no headlines available
# Setup: Mock data source to return empty headline list
# Assert: collector returns {}

# Test: synthesis handles {} sentiment without error
# Setup: Call RuleBasedSynthesizer.synthesize with sentiment={}
# Assert: no KeyError, synthesis completes, sentiment contribution is 0.0

# Test: synthesis handles None sentiment without error
# Setup: Call synthesize with sentiment=None
# Assert: no error, same behavior as {}
```

---

## Section 6: Bind All Docker Services to Localhost (Item 0.6)

No automated tests. Manual validation:

```
# Verify: docker-compose.yml has 127.0.0.1 prefix on all port bindings
# Verify: docker compose up starts successfully
# Verify: services accessible from localhost but not from other machines
# Verify: inter-container communication still works (graphs can reach postgres, ollama, langfuse)
```

---

## Section 7: Remove All Default Passwords (Item 0.7)

**Test file:** `tests/unit/test_startup_validation.py`

```python
# Test: start.sh rejects missing POSTGRES_PASSWORD
# Setup: Run start.sh with POSTGRES_PASSWORD unset
# Assert: exit code 1, error message mentions POSTGRES_PASSWORD

# Test: start.sh rejects default password "quantstack"
# Setup: Run start.sh with POSTGRES_PASSWORD=quantstack
# Assert: exit code 1, error message mentions default value

# Test: start.sh rejects short passwords (<12 chars)
# Setup: Run start.sh with POSTGRES_PASSWORD=short
# Assert: exit code 1, error message mentions minimum length

# Test: start.sh accepts valid passwords
# Setup: Run start.sh with all passwords set to 12+ char non-default values
# Assert: passes validation (may fail later on docker, but password check passes)

# Test: start.sh rejects default LANGFUSE_DB_PASSWORD
# Test: start.sh rejects default LANGFUSE_INIT_USER_PASSWORD
```

Manual validation:
```
# Verify: docker-compose.yml has no :-default fallbacks for passwords
# Verify: docker compose up without .env shows missing variable warnings
```
