# Integration Notes — Opus Review Feedback

## Integrating (9 items)

### 1. P0: Safety check fallback is fail-OPEN
**Integrating.** This is the most critical finding. `{"halted": False}` as a parse fallback for the safety_check agent means parse failure = safety bypass. Will add to Section 4: ALL safety-critical agent fallbacks must fail CLOSED. Specifically: safety_check → `{"halted": True}`, risk assessment failures → reject (not approve).

### 2. P0: Unprotected window in contingent SL path
**Integrating.** Will specify: fill detection via Alpaca trade update WebSocket stream (already used by execution monitor), max 2s latency between fill and SL submission, 3 retries with exponential backoff on SL submission failure, and if all retries fail → trigger kill switch for that symbol.

### 3. P1: Enumerate all 15 psycopg2 import sites
**Integrating.** Will replace the vague "grep for imports" risk with a concrete migration checklist of all files. This is necessary for a credible migration plan.

### 4. P1: Blocklist → allowlist for prompt sanitization
**Integrating.** Will restructure Section 3 to make field-level extraction (allowlist) the PRIMARY defense, with blocklist-based sanitization as a secondary monitoring/alerting signal (not a security boundary).

### 5. Checkpoint data retention policy
**Integrating.** Will add retention strategy: keep last 48 hours of checkpoint data, prune older rows via a scheduled job.

### 6. Safety check + all fallback value audit
**Integrating.** Will add to Section 4: systematic audit of every `parse_json_response` fallback value across all three graphs, ensuring safety-critical paths fail closed.

### 7. WAL archive retention
**Integrating.** Will add 7-day WAL archive pruning to Section 8.

### 8. EventBus publication from kill_switch must be best-effort
**Integrating.** Will specify try/except around EventBus publish in kill_switch.trigger() — cannot delay or prevent kill switch activation.

### 9. Injection attempt monitoring/alerting
**Integrating.** Will add to Section 3: log and alert when sanitization strips content, as an anomaly detection signal.

## NOT Integrating (5 items)

### 1. E*Trade scope creep
**Not integrating (disagree).** The user explicitly chose "All brokers" in the interview. The plan should follow user intent. E*Trade adapter gets the interface; implementation can be thin/stub for Phase 1.

### 2. psycopg2 rollback plan / feature flag
**Not integrating.** A feature flag for the DB driver adds significant complexity for an unlikely scenario. The existing test suite is the safety net. If tests pass, the migration is sound.

### 3. Read-only root filesystem
**Not integrating for Phase 1.** Good hardening but adds debugging friction and complexity with volume mounts. Phase 2 candidate.

### 4. Flask → http.server for scheduler health
**Not integrating.** The plan doesn't specify Flask — it says "lightweight HTTP health endpoint" which can be http.server or a simple socket. This is an implementation detail, not a plan-level concern.

### 5. Deadlock ordering constraint
**Partially integrating.** The review's concern is valid but the mitigation is already implicit: position updates are always single-row (one symbol per transaction). Will make this explicit in the plan rather than adding a full deadlock-ordering protocol.

### 6. `client_order_id` collision
**Integrating via minor fix.** Will specify millisecond precision + short random suffix.

### 7. Migration sequencing (research graph first for prompt injection)
**Integrating.** Good point — research graph has highest untrusted data exposure. Will reverse the migration order.

### 8. Partial fill + SL rejection edge case
**Integrating.** Will add to Section 2.

### 9. init process for containers
**Integrating.** Trivial addition (`init: true` in docker-compose), prevents zombie processes.
