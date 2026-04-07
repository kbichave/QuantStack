# Implementation Findings

## Technical Decisions
| Decision | Rationale | Section |
|----------|-----------|---------|
| Created `_DictRow` class instead of plain `dict_row` | psycopg3's `dict_row` breaks tuple unpacking (`for a, b in rows:`) used extensively in EventBus, coordination, etc. `_DictRow` supports both dict access and positional/tuple unpacking | section-01 |
| `set_json_loads` with bytes decode | psycopg3 passes bytes (not str) to JSON loader; added `decode("utf-8")` wrapper to return raw strings matching psycopg2 behavior | section-01 |
| `_execute_values` helper in pg_storage.py | Replaces `psycopg2.extras.execute_values` with `executemany` (psycopg3 pipeline mode); SQL template expansion from `VALUES %s` to per-row placeholders | section-01 |
| `min_size = min(4, max_size)` in pool | Prevents ValueError when PG_POOL_MAX < 4 | section-01 |
| safety_check fallback defaults to `halted: True` | P0 fail-CLOSED: parse failure must halt, never proceed. This is the single most critical safety invariant | section-02, section-04 |
| `stop_price: float` (not Optional) on BracketIntent | Type-level enforcement — impossible to construct an entry order without a stop-loss | section-02 |
| XML-tagged safe_prompt() over f-strings | Prevents prompt injection by wrapping external data in XML tags and stripping tags from values | section-03 |
| 21 Pydantic output models with DLQ | Schema validation catches malformed LLM output; dead letter queue preserves evidence for debugging | section-04 |
| PostgresSaver with dedicated pool (min=2, max=6) | Separate from main app pool to prevent checkpoint writes from starving query paths | section-06 |
| Best-effort EventBus publish in kill_switch | EventBus failure must NEVER prevent kill switch activation — sentinel file is the authoritative signal | section-07 |
| Single-row SELECT FOR UPDATE (no multi-row locks) | Eliminates deadlock risk entirely — deadlock requires 2+ rows locked in different order | section-10 |
| FaultyBroker wrapper for chaos testing | Configurable failure injection (fail_next_n, fail_on method, custom error) with full call log | section-11 |

## Issues Encountered
| Issue | Section | Resolution |
| Mock patches targeting wrong import path | section-06, section-10 | Deferred imports inside functions require patching the source module (e.g., `langgraph.checkpoint.postgres.PostgresSaver`) not the importing module |
| execute_trade catches Exception broadly | section-02 | Tests check return dict `{"success": False, "error": "...stop_price..."}` instead of expecting ValueError |
| Edit tool uniqueness error | section-04 | Provided more surrounding context to disambiguate duplicate strings |
| Docker integration tests fail without daemon | section-05 | Expected — tagged `@pytest.mark.integration`, skip when Docker not running |
|-------|---------|------------|

## Resources
- Planning dir: /Users/kshitijbichave/Personal/Trader/CTO_ONBOARDING_AUDIT/specs/sessions/5c8928af
- Test command: uv run pytest
