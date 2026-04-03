# Interview Transcript

## Q1: Primary LLM Provider
**Question:** Which LLM provider should be the PRIMARY default?
**Answer:** AWS Bedrock — current config, uses IAM, best for cost control.

## Q2: Process Management
**Question:** For process management when on vacation, which approach?
**Answer:** Docker Compose — containerized with health checks, auto-restart, includes Ollama + ChromaDB + Langfuse as services.

## Q3: Langfuse Hosting
**Question:** Self-hosted or cloud-hosted Langfuse?
**Answer:** Self-hosted via Docker — full data privacy, no strategy data leaves the machine.

## Q4: LLM Failover
**Question:** What should happen when the primary LLM provider (Bedrock) fails?
**Answer:** Auto-fallback chain: Bedrock → Anthropic → OpenAI → Ollama.

## Q5: RAG Knowledge Base Sources
**Question:** What documents should be ingested for RAG?
**Answer:** All of the above — trade outcomes + reflexion episodes, strategy registry + workshop lessons, research papers + market reports.

## Q6: Alerting
**Question:** What notification channel for alerting?
**Answer:** Just logs. The system should be self-healing — no push notifications needed.

## Q7: Self-Healing Scope
**Question:** What should the system auto-recover from?
**Answer:** Everything possible — LLM failover, API retries, stuck agent restart, crashed crew restart, stale data refresh, DB reconnect, Ollama restart.

## Q8: Strategy Promotion Autonomy
**Question:** Should the system auto-promote strategies?
**Answer:** Full automation but with LLM-based reasoning. No human-added rules or magic numbers derived from thin air. The LLM should reason about promotion decisions based on evidence, not hardcoded gates.

## Q9: Cost Budget
**Question:** Budget tolerance for LLM API costs per day?
**Answer:** No hard limit, optimize later. Focus on getting it working first.

## Q10: LLM vs Rules (Thresholds)
**Question:** Should LLM-based reasoning replace ALL numeric thresholds?
**Answer:** Full LLM reasoning, no hardcoded thresholds anywhere. Even risk limits should be LLM-determined based on market conditions and portfolio state. Maximum autonomy.

## Q11: Migration Approach
**Question:** Big-bang cutover or gradual strangler-fig migration?
**Answer:** Big-bang — build the CrewAI system, switch over completely. Don't try to run both in parallel.

## Q12: Capital Allocation
**Question:** Maximum dollar amount for paper trading?
**Answer:** $25K total — $20K for equity, $5K for options. Full portfolio is $250K but only allocating $25K to this system.

## Q13: Memory Migration
**Question:** How to handle existing .claude/memory/ files?
**Answer:** Ingest into RAG at startup, then use CrewAI memory going forward. One-time import of historical knowledge.

## Q14: Multi-LLM Provider Support (User-initiated requirement)
**User stated:** System should support Gemini, OpenAI, Claude API keys, and Bedrock for LLM providers. Select appropriate default models for each agent across all providers.
