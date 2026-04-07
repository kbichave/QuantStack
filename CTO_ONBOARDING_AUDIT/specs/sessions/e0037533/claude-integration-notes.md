# Integration Notes — Opus Review Feedback

## Integrating

1. **[Blocking] Audit node return keys before Pydantic migration** — Integrating. The `alpha_signals` ghost field is a real find. Adding a prerequisite step to Section 1 to dynamically audit all node returns before writing models.

2. **[Blocking] Include SymbolValidationState** — Integrating. Need to verify this exists in state.py, but if it does, it must be in scope.

3. **[High] Move DB migration to step 1** — Integrating. Correct — DB migrations are additive/zero-risk and unblock circuit breaker + DLQ testing. Reordering dependency chain.

4. **[High] Minimum stop distance floor** — Integrating. Adding 2x ATR floor (or 1% of price, whichever is larger) to prevent stops within noise range.

5. **[High] Handle stop_price = None** — Integrating. If no stop exists, regime flip can't tighten — should SET a stop at the floor distance instead.

6. **[Medium] Increase circuit breaker cooldown to 300s** — Integrating. 30s is too short for cycle intervals of 60-300s. Default 300s with per-node configurability.

7. **[Medium] Timeout + fallback for Haiku summarization** — Integrating. 2s hard timeout, fallback to truncation. Prefer pre-computing at merge points.

8. **[Medium] Atomic increment for circuit breaker** — Integrating. `UPDATE ... SET failure_count = failure_count + 1 RETURNING failure_count` prevents read-modify-write race.

9. **[Medium] Heat budget: per-graph vs system-wide** — Integrating. Must be system-wide (all 3 graph services deploy capital from the same portfolio). Query DB for today's deployments, no in-memory cache.

10. **[Low] SQL placeholder style** — Integrating. Match codebase convention (`?` placeholders).

11. **[Low] CLAUDE.md as first commit** — Integrating. Sequence it explicitly in dependency order.

12. **[Medium] execute_exits blocking tradeoff** — Integrating as an explicit acknowledgment in Section 2.

13. **Missing tests** — Integrating all 6 test gaps: ghost field audit, concurrent breaker updates, summarization failure, stop_price=None, tool access bypass, circuit breaker + execution gate interaction.

## NOT Integrating

14. **[Low] Per-node field allow-list instead of output models** — NOT integrating. The user explicitly chose "typed output models per node" in the interview (Q12). While the allow-list alternative reduces boilerplate, the user made a deliberate architectural decision for maximum type safety. Respecting that choice. The boilerplate cost is real but manageable with code generation or base class patterns.

15. **Execution gate as node vs conditional edge** — NOT integrating as a change. The reviewer raises a valid concern about conditional edges not seeing execution metadata. However, the execution gate only needs to inspect the `errors` list in state (which is populated by node error handling, not execution metadata). Errors from both caught exceptions and returned error states end up in the same `errors` list. The conditional edge approach is sufficient for this gate. Will add a note clarifying this.
