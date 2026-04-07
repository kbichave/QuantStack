# Section 11: Priority-Based Message Pruning

## Problem

When conversation history exceeds 150K chars (~37k tokens), the `_prune_messages()` function in `src/quantstack/graphs/agent_executor.py` drops the oldest tool round pairs using FIFO ordering. This means late-pipeline agents like `fund_manager` (10th agent in the trading graph) can lose critical upstream results -- for example, `position_review` output that the agent needs for allocation decisions. FIFO pruning treats all messages as equally disposable, which they are not.

## Dependencies

- **section-03-pydantic-state-migration**: Message priority tags are stored as typed metadata fields in the Pydantic state models. The Pydantic migration must be complete before priority metadata is available on messages.
- **Agent config**: `agents.yaml` files for each graph must have `priority_tier` fields added per agent.

## Design

### Hybrid Priority System

Priority is assigned through two mechanisms that combine at message construction time:

1. **Config defaults**: Each agent in `agents.yaml` gets a `priority_tier` field (`P0`, `P1`, `P2`, `P3`). This is the default priority for all messages originating from that agent.
2. **Type overrides**: Certain message types are always `P0` regardless of which agent produced them:
   - Risk gate output
   - Kill switch status
   - Position state / portfolio context
   - Error messages from blocking nodes (as classified in section-05)

Type overrides take precedence over config defaults. A `P2` agent producing a risk gate output message gets that message tagged `P0`.

### Priority Tiers

| Tier | Behavior | Examples |
|------|----------|----------|
| P0 | Never pruned, never summarized | Risk/execution state, kill switch, position state, blocking-node errors |
| P1 | Summarized (not dropped) when P2 exhausted and still over budget | Signal briefs, trade decisions, regime assessments |
| P2 | Pruned first (oldest-first within tier) | Raw analysis text, verbose tool outputs, intermediate reasoning |
| P3 | Never added to LLM context | Debug logs, trace metadata -- excluded at construction time |

### Pruning Algorithm

Replaces the current FIFO logic in `_prune_messages()`:

1. Calculate current message budget usage against the 150K char threshold.
2. If over budget, remove all P3 messages (these should already be excluded, but sweep as safety).
3. If still over budget, prune P2 messages oldest-first.
4. If still over budget, summarize P1 messages using Haiku (cheap/fast model). Replace verbose content with a condensed summary. Hard timeout of 2 seconds on the summarization call. If Haiku is slow or unavailable, fall back to truncation (first N chars of the original content) rather than stalling the pipeline.
5. P0 messages are never pruned or summarized under any circumstances.

### Merge-Point Compaction

Prefer pre-computing summaries at merge points over lazy summarization during pruning. At `merge_parallel` and `merge_pre_execution`, summarize verbose branch outputs before passing them downstream. This runs once per branch merge, while pruning-time summarization would run before every subsequent agent invocation.

Use LangGraph's `RemoveMessage` with the `add_messages` reducer for rolling-window compaction: remove old verbose messages, insert a summary message in their place.

## Tests

All tests belong in `tests/unit/test_message_pruning.py`.

```python
# Test: P2 messages pruned before P1 when over budget
# Construct a message list with mixed P0/P1/P2 messages that exceeds the char budget.
# After pruning, P2 messages should be removed first while P0 and P1 remain intact.

# Test: P1 messages summarized (not pruned) when P2 exhausted and still over budget
# Construct a message list where P2 removal alone is insufficient to get under budget.
# P1 messages should be replaced with shorter summaries, not removed entirely.

# Test: P0 messages never pruned or summarized regardless of budget
# Construct a message list that is over budget even after removing all P2 and summarizing all P1.
# P0 messages must remain untouched -- their full content preserved.

# Test: P3 messages never added to LLM context
# Tag messages as P3 and verify they are excluded from the message list passed to the LLM.

# Test: type override -- risk gate output is P0 even if source agent defaults to P1
# Create a message from a P1-configured agent that contains risk gate output.
# Verify the message is tagged P0 and survives pruning.

# Test: type override -- error from blocking node is P0
# Create an error message from a blocking node whose agent config is P2.
# Verify the message is tagged P0.

# Test: Haiku summarization timeout (>2s) falls back to truncation
# Mock the Haiku LLM call to exceed the 2-second timeout.
# Verify the pruner falls back to truncation (first N chars) and the pipeline continues.

# Test: Haiku unavailable falls back to truncation
# Mock the Haiku LLM call to raise a connection error.
# Verify truncation fallback is used without raising an exception.

# Test: message priority tag correctly set in metadata during construction
# Create messages through the normal message construction path with agent config priority.
# Verify each message's metadata contains the correct priority_tier value.

# Test: compaction at merge point -- verbose branch outputs replaced with summary
# Simulate a merge_parallel step with verbose branch outputs.
# Verify the outputs are replaced with a summary message and the original verbose
# messages are removed via RemoveMessage.
```

## Implementation Details

### Files to Modify

- **`src/quantstack/graphs/agent_executor.py`**: Replace the existing `_prune_messages()` function with the priority-aware algorithm. Add priority tag reading from message metadata. Add Haiku summarization call with 2-second timeout and truncation fallback.
- **`src/quantstack/graphs/*/config/agents.yaml`**: Add `priority_tier` field (P0/P1/P2/P3) to each agent definition in all three graph configs (trading, research, supervisor).

### Message Construction

When messages are constructed (wherever tool results, agent outputs, or system messages are added to the conversation), attach a `priority_tier` key in the message's metadata dict. The value comes from:

1. Check if the message matches a type override condition (risk gate output, kill switch, position state, blocking-node error). If yes, set `P0`.
2. Otherwise, use the agent's configured `priority_tier` from `agents.yaml`.

This tagging happens at construction time, not at pruning time, so the pruner only needs to read metadata -- it does not need to re-classify messages.

### Haiku Summarization

The summarization call should use the cheapest/fastest available model (Haiku tier). The prompt should be minimal: "Summarize the following agent output in 2-3 sentences, preserving any numerical values, ticker symbols, and directional signals." The call must have a strict 2-second timeout. On any failure (timeout, rate limit, connection error, parse failure), fall back to truncating the original message to a fixed character limit (e.g., first 500 chars with a `[truncated]` suffix).

### Merge-Point Compaction

At `merge_parallel` and `merge_pre_execution` nodes in the trading graph, add compaction logic that:

1. Collects all messages produced by the parallel branches.
2. Summarizes them into a single compact message per branch.
3. Uses `RemoveMessage` to delete the verbose originals from state.
4. Inserts the summary messages via the `add_messages` reducer.

This reduces the message volume before downstream agents process the merged state, making the per-agent pruning step less aggressive.

### Agent Priority Assignments (Suggested Defaults)

These go in the respective `agents.yaml` files as a `priority_tier` field on each agent:

**Trading graph agents:**
- `safety_check`, `position_review`, `execute_exits`, `risk_sizing`: P0 (risk/execution)
- `plan_day`, `fund_manager`, `entry_scan`: P1 (decision-making)
- `market_intel`, `earnings_analysis`, `trade_reflector`, `reflect`: P2 (analysis/reasoning)

**Research graph agents:**
- Assign based on the same principle: anything feeding risk or execution is P0, strategy discovery is P1, verbose analysis is P2.

**Supervisor graph agents:**
- Health monitoring outputs: P1
- Diagnostic verbose outputs: P2

The exact assignments should be validated against actual message volumes during paper trading. These are starting defaults.
