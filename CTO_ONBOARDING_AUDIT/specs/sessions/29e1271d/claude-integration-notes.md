# Integration Notes — Opus Review Feedback

## Integrating

| Issue | Action |
|-------|--------|
| **C1: TradingState extra=forbid** | Integrating. Add explicit `cycle_attribution: dict = {}` field to TradingState in Section 4. Critical catch — would have crashed at runtime. |
| **C2: ACK timeout uses consumer interval** | Integrating. Change to fixed duration per event type (600s for risk events). Document NULL-safety for existing rows. |
| **M1: factor_exposure_history table missing** | Integrating. Add to Section 1 schema definitions. |
| **M2: cycle_attribution table missing** | Integrating. Add to Section 1 schema definitions. |
| **M3: Graph routing mechanism unspecified** | Integrating. Specify conditional edge after `safety_check` that routes based on `get_operating_mode()`. |
| **M4: Order side conflation** | Integrating. Check absolute exposure change, not order side. Critical for short positions. |
| **M5: EDGAR CIK mapping** | Integrating. Use SEC `company_tickers.json` endpoint with caching. |
| **M6: No tests** | Integrating. Tests will be covered in the TDD plan (Step 16) — that's the designated place for test specifications. |
| **m3: Discord TODO trigger** | Integrating. Use env var trigger condition. |
| **m4: Semaphore vs rate limiter** | Integrating. Use both: semaphore for concurrency + existing AV rate limiter for quota. |
| **m5: split_adjustments constraint** | Integrating. Add event_type to unique constraint. |
| **S1: Section 5 to Batch 1** | Integrating. Move system alerts to Batch 1 alongside schema — it's foundational. |
| **S3: Alpaca double-adjustment** | Integrating. Add reconciliation check before applying split adjustments. |

## NOT Integrating

| Issue | Reason |
|-------|--------|
| **m1: edgartools maturity** | User explicitly chose full AV+EDGAR coverage. Verify during implementation — not a plan change. |
| **m2: LLM health probe** | Good optimization but over-engineering for Phase 9. The probe is simple and cheap. Can optimize later. |
| **S2: Fan-out rollback plan** | The env var override is already the rollback — this is self-evident and doesn't need plan text. |
| **S4: factor_config as typed model** | Key-value table is simpler and sufficient for 4 config keys. Type validation at read time is fine for this scale. Premature abstraction for 4 rows. |
