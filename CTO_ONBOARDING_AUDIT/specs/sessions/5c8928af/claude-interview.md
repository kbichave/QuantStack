# Interview Transcript — Phase 1 Safety Hardening

## Q1: Broker scope for bracket order hardening
**Question:** The spec says bracket order failure should place a separate contingent SL, but the current alpaca_broker.py falls back to a plain order. Should we also retrofit existing brokers (E*Trade) with the same bracket-or-contingent-SL pattern, or is Alpaca the only broker for now?

**Answer:** All brokers — retrofit E*Trade and PaperBroker with the same bracket/contingent pattern.

## Q2: Startup reconciliation behavior
**Question:** For startup reconciliation (verifying all positions have stops): should the system refuse to trade until reconciliation passes, or submit missing stops automatically and continue?

**Answer:** Auto-fix and continue — automatically submit missing stop orders, log the event, and proceed.

## Q3: Prompt injection defense depth
**Question:** Should we also implement the Dual LLM pattern (research LLM can't access execution tools), or is structured templates + sanitization sufficient for Phase 1?

**Answer:** Templates + Dual LLM separation — replace f-strings with XML-tagged templates, add sanitization, AND enforce that research-facing LLMs never see execution tools.

## Q4: Database driver strategy for PostgresSaver
**Question:** The codebase uses psycopg2 (ThreadedConnectionPool) but PostgresSaver requires psycopg3. Should we migrate the entire db.py to psycopg3, or run two connection systems in parallel?

**Answer:** Migrate to psycopg3 — bigger change but eliminates maintaining two drivers.

## Q5: Output schema design for 21 agents
**Question:** Should we define one Pydantic model per agent, or group agents by output shape?

**Answer:** One model per agent — 21 distinct Pydantic models. Maximum type safety, easier to evolve independently.

## Q6: Backup storage target
**Question:** Do you have an AWS account/bucket ready for offsite backups?

**Answer:** Local + WAL for now — start with local pg_dump + WAL archiving, add offsite later.

## Q7: Concurrent position access patterns
**Question:** Are there concurrent writers beyond execution monitor + trading graph?

**Answer:** Just execution monitor + trading graph. Supervisor only reads.

## Q8: Scheduler ibkr_mcp import chain
**Question:** Should the plan fix the ibkr_mcp import chain properly or use a lazy import workaround?

**Answer:** Fix the import chain properly — remove or isolate the ibkr_mcp dependency so scheduler imports cleanly.

## Q9: Migration risk for existing positions
**Question:** How many positions are typically open, and could legacy positions lack ATR data?

**Answer:** No open positions currently — system is in paper mode / fresh start, migration is a no-op.

## Q10: EventBus poll frequency
**Question:** Should all three graphs poll events at cycle start only, or also mid-cycle before critical actions?

**Answer:** Start + before execution — poll before execute_entries and execute_exits to catch events mid-cycle.
