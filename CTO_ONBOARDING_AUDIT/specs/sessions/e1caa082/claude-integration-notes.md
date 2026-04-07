# Integration Notes — Opus Review Feedback

## Integrating

| Issue | Action |
|-------|--------|
| **P0: `extra="forbid"` state model incompatibility** | Add migration note to Section 3. Clean restart of graph services required on deployment. Verify no checkpoint schema conflicts. |
| **P0: Embedding provider unspecified** | Specify Amazon Titan embeddings via Bedrock (already the primary provider) in Section 7. Falls back to local sentence-transformers if Bedrock unavailable. |
| **P0: pgvector not in Docker image** | Add to Section 13 (Database Migrations). Switch to `pgvector/pgvector:pg16` image in docker-compose.yml. |
| **P1: Overnight runner crash recovery** | Add crash recovery design to Section 4: DB-persisted budget, unconditional morning validator, idempotent restart. |
| **P1: Mandate failure fallback** | Add conservative default mandate to Section 10. max_new_positions=0 on CIO failure. |
| **P1: AR-9/AR-1 budget interaction** | Clarify in Sections 3 and 4: per-cycle budgets for daytime research, per-experiment budget for overnight, both scoped independently. |
| **P1: Prompt caching section missing** | Add new Section 14 covering prompt caching implementation in LLM provider layer. |
| **P2: Feature enumeration cap** | Add 2000 candidate hard cap to Section 5. |
| **P2: 5-minute experiment timeout, not sleep** | Clarify in Section 4: run back-to-back with 5-min timeout per experiment. |
| **P2: Governance before meta agents in 10D** | Reorder Section 10 before Section 9 in implementation sequence note. |
| **P2: Validation milestones** | Add acceptance gate tests to testing strategy section. |
| **P3: Consensus feature flag** | Add CONSENSUS_ENABLED env var to Section 8. |
| **P3: Threshold extraction prerequisite** | Add note to Section 9: extract hardcoded thresholds to config as prerequisite. |

## NOT Integrating

| Issue | Reason |
|-------|--------|
| **DSPy spike/PoC** | Valid concern but the plan already notes DSPy as the approach. The spike happens naturally during 10C/10D implementation — the meta_prompt_optimizer section is the last to implement. If DSPy proves unworkable, fall back to manual A/B testing of prompt variants (same logic, different optimization engine). Not worth adding a separate spike task. |
| **Consensus simpler alternative** | The spec explicitly requires the 3-agent consensus pattern. The feature flag addresses the concern about unproven value. |
| **Event schema inline vs. separate file** | Minor implementation detail. Leaving as-is — the implementer can decide inline vs. file. |
| **Cold-start prioritization formula** | Valid observation. The formula degenerates to "cheap experiments in current regime" early on — which is actually a reasonable heuristic for a new system. Will add an acknowledgment but not change the formula. |
| **KG schema in 10B** | Would add complexity to 10B for marginal benefit. The backfill from autoresearch_experiments is straightforward and bounded (~3000 rows). Not worth pulling forward. |
| **Monitoring/alerting for new processes** | The supervisor graph already monitors health. New processes register via event bus heartbeats (LOOP_HEARTBEAT). This is implicit in the existing architecture. |
| **Rollback plan per sub-phase** | Each sub-phase is additive. The existing system continues to work if new components are disabled. Feature flags (consensus, meta agents) provide the rollback mechanism for the riskier components. |
