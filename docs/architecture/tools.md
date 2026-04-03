# Tool Layer Architecture

## Overview

QuantStack agents interact with markets, data, and internal systems through a
structured tool layer. Tools are the boundary between LLM reasoning and
deterministic computation -- every market data fetch, risk calculation, order
execution, and model training call passes through this layer.

The tool layer lives under `src/quantstack/tools/` and is organized into two
sub-packages plus shared infrastructure:

- **`tools/langchain/`** -- LLM-facing tools decorated with `@tool`, callable by
  agents during graph execution.
- **`tools/functions/`** -- Deterministic Python functions called directly by
  graph node code. Not exposed to LLMs.
- **`tools/_shared.py`** -- Shared implementation logic used by both tiers
  (e.g., `run_backtest_impl`, `register_strategy_impl`).
- **`tools/_state.py`** -- `TradingContext` management (`require_ctx()`,
  `get_ctx()`). Provides DB connections, broker handles, and system state.
- **`tools/_helpers.py`** -- Common utilities (JSON serialization, error formatting).
- **`tools/models.py`** -- Shared Pydantic models (e.g., `StrategyRecord`).

---

## Two-Tier Architecture

The tool layer enforces a strict separation between LLM-callable tools and
internal computation:

```
                    Agent (LLM)
                        |
                        v
            +------------------------+
            |   LLM-Facing Tools     |  <-- @tool decorated, string in / JSON out
            |   tools/langchain/     |      Stateless, schema-validated
            +------------------------+
                        |
                        v
            +------------------------+
            |   Deterministic Funcs  |  <-- Pure Python, called by graph nodes
            |   tools/functions/     |      No LLM interaction
            +------------------------+
                        |
                        v
            +------------------------+
            |   Shared Infra         |  <-- _shared.py, _state.py, models.py
            |   Core Libraries       |      Direct Python imports
            +------------------------+
```

**Why two tiers?**

- LLM-facing tools need string serialization, input validation, and descriptive
  docstrings that serve as the tool's "API docs" for the model. They carry
  overhead that internal code does not need.
- Deterministic functions are called on hot paths (risk checks, position sizing)
  where the overhead of string serialization would be wasteful and where LLM
  discretion is not wanted.
- Keeping them separate makes it clear which functions an agent can invoke
  versus which are internal implementation details.

---

## Tool Registry

**File:** `src/quantstack/tools/registry.py`

The registry maps string names to LangChain `BaseTool` objects. Agent YAML
configs (`src/quantstack/graphs/*/config/agents.yaml`) reference tools by
string name; the registry resolves them at graph construction time.

```python
TOOL_REGISTRY = {
    "signal_brief": signal_brief,
    "multi_signal_brief": multi_signal_brief,
    "fetch_market_data": fetch_market_data,
    "fetch_fundamentals": fetch_fundamentals,
    "fetch_earnings_data": fetch_earnings_data,
    "fetch_portfolio": fetch_portfolio,
    "fetch_options_chain": fetch_options_chain,
    "compute_greeks": compute_greeks,
    "compute_risk_metrics": compute_risk_metrics,
    "compute_position_size": compute_position_size,
    "execute_order": execute_order,
    "run_backtest": run_backtest,
    "train_model": train_model,
    "compute_features": compute_features,
    "web_search": web_search,
    "search_knowledge_base": search_knowledge_base,
    "fetch_strategy_registry": fetch_strategy_registry,
    "check_system_status": check_system_status,
    "check_heartbeat": check_heartbeat,
}
```

### Resolution

`get_tools_for_agent(tool_names: list[str]) -> list[BaseTool]` resolves a list
of string names to tool objects. On a missing name, it raises `KeyError` with
the full list of available tool names -- this makes YAML typos immediately
visible at startup rather than at runtime.

### Design constraints

- The registry is the **single source of truth** for which tools exist.
- Supervisor tools (`check_system_status`, `check_heartbeat`) are defined
  inline in `registry.py` rather than in a separate file because they are
  thin wrappers with no shared logic.
- Adding a tool to a file under `tools/langchain/` does **not** make it
  available to agents until it is also registered here.

---

## LLM-Facing Tool Catalog

All LLM-facing tools live under `src/quantstack/tools/langchain/`. Each is
decorated with `@tool`, accepts string arguments, and returns a JSON string.
Tools are stateless -- all persistent state lives in PostgreSQL.

| Tool | File | Description |
|------|------|-------------|
| `signal_brief` | `signal_tools.py` | Generate a trading signal summary for a single ticker |
| `multi_signal_brief` | `signal_tools.py` | Generate signal summaries for multiple tickers in one call |
| `fetch_market_data` | `data_tools.py` | Retrieve price/volume data (daily, intraday) from Alpha Vantage or Alpaca |
| `fetch_fundamentals` | `data_tools.py` | Retrieve fundamental data (balance sheet, income statement, ratios) |
| `fetch_earnings_data` | `data_tools.py` | Retrieve earnings history, surprises, and upcoming dates |
| `fetch_portfolio` | `portfolio_tools.py` | Get current portfolio positions, P&L, and allocation from Alpaca |
| `fetch_options_chain` | `options_tools.py` | Retrieve options chain for a ticker (calls, puts, strikes, expiries) |
| `compute_greeks` | `options_tools.py` | Calculate option Greeks (delta, gamma, theta, vega, rho) |
| `compute_risk_metrics` | `risk_tools.py` | Calculate VaR, max drawdown, Sharpe, and other risk metrics |
| `compute_position_size` | `risk_tools.py` | Determine position size given risk budget, stop loss, and portfolio state |
| `execute_order` | `execution_tools.py` | Submit an order to Alpaca (paper or live, gated by risk checks) |
| `run_backtest` | `backtest_tools.py` | Run a historical backtest for a given strategy and ticker |
| `train_model` | `ml_tools.py` | Train an ML model (XGBoost, LSTM, etc.) on computed features |
| `compute_features` | `ml_tools.py` | Generate feature matrix for a ticker (technical, fundamental, sentiment) |
| `web_search` | `intelligence_tools.py` | Search the web for news, filings, or market commentary |
| `search_knowledge_base` | `learning_tools.py` | Query the internal knowledge base (pgvector similarity search) |
| `fetch_strategy_registry` | `strategy_tools.py` | Retrieve registered strategies, their status, and performance history |
| `check_system_status` | `registry.py` (inline) | Check overall system health and kill-switch state |
| `check_heartbeat` | `registry.py` (inline) | Verify service liveness for a specific graph |

### Tool contract

Every LLM-facing tool follows the same contract:

1. **Input:** A single string argument (or structured args via LangChain schema).
2. **Output:** A JSON string. On success, contains the result payload. On
   failure, contains an `{"error": "..."}` object with a human-readable message.
3. **Side effects:** Only `execute_order` has side effects (placing trades).
   All other tools are read-only.
4. **Statefulness:** None. Tools do not cache results or maintain internal state
   between calls.

---

## Deterministic Functions

**Directory:** `src/quantstack/tools/functions/`

These are plain Python functions called directly by graph node code. They are
not decorated with `@tool` and are not visible to LLMs. Use these when:

- The operation must always execute (no LLM discretion).
- Performance matters (no serialization overhead).
- The logic is an internal implementation detail.

| File | Purpose |
|------|---------|
| `data_functions.py` | Data processing utilities -- normalization, resampling, gap filling |
| `execution_functions.py` | Order routing, fill handling, execution quality tracking |
| `risk_functions.py` | Risk calculations -- VaR, drawdown, correlation, exposure limits |
| `system_functions.py` | System health checks, structured logging, monitoring helpers |

### When to use functions vs. LLM-facing tools

Use a **function** when the graph node always calls it unconditionally and the
LLM should not decide whether to call it. Risk gate checks and order routing
are good examples -- these are mandatory, not optional.

Use an **LLM-facing tool** when the agent needs to decide at runtime whether
and how to call it. Data fetching and analysis are good examples -- the agent
chooses which tickers to research and which tools to use.

---

## Shared Infrastructure

**Directory:** `src/quantstack/tools/`

Files at the top level of `tools/` provide shared logic used by both tiers:

| File | Purpose |
|------|---------|
| `_shared.py` | Implementation logic shared across tiers (backtest, strategy CRUD, portfolio snapshot) |
| `_state.py` | `TradingContext` lifecycle -- `require_ctx()` returns a context with DB pool, broker handle, and system state |
| `_helpers.py` | JSON serialization, error formatting, common utilities |
| `models.py` | Pydantic models shared across tool files (e.g., `StrategyRecord`) |
| `registry.py` | `TOOL_REGISTRY` dict + `get_tools_for_agent()` resolver |

`TradingContext` (from `_state.py`) is the primary mechanism for accessing
runtime resources. Both LLM-facing tools and deterministic functions use
`require_ctx()` to get database connections, broker handles, and system state
without passing them as arguments through the call chain.

---

## Adding a New Tool

### Step 1: Decide which tier

- Will an LLM choose when to call this? -> `tools/langchain/`
- Is it always called by graph node logic? -> `tools/functions/`

### Step 2: Implement the function

**For LLM-facing tools** (`tools/langchain/`):

Create or extend a file in `src/quantstack/tools/langchain/`. Use the `@tool`
decorator from LangChain. The function docstring becomes the tool description
the LLM sees -- write it carefully.

```python
# src/quantstack/tools/langchain/example_tools.py
from langchain_core.tools import tool
import json

@tool
def my_new_tool(query: str) -> str:
    """One-line description the LLM will see.

    Use this tool when you need to [specific use case].
    Input should be [describe expected format].
    """
    # Implementation here
    result = do_something(query)
    return json.dumps(result)
```

**For deterministic functions** (`tools/functions/`):

Create or extend a file in `src/quantstack/tools/functions/`. No decorator
needed. Type-hint inputs and outputs.

```python
# src/quantstack/tools/functions/example_functions.py
def my_internal_function(data: dict, threshold: float) -> dict:
    """Compute something. Called by graph nodes, not by LLMs."""
    # Implementation here
    return {"result": computed_value}
```

### Step 3: Register (LLM-facing tools only)

Add the tool to `TOOL_REGISTRY` in `src/quantstack/tools/registry.py`:

```python
from quantstack.tools.langchain.example_tools import my_new_tool

TOOL_REGISTRY = {
    # ... existing tools ...
    "my_new_tool": my_new_tool,
}
```

### Step 4: Wire to agents (LLM-facing tools only)

Add the tool name to the relevant agent's `tools:` list in the YAML config:

```yaml
# src/quantstack/graphs/research/config/agents.yaml
quant_researcher:
  tools:
    - fetch_market_data
    - my_new_tool       # <-- add here
```

### Step 5: Test

- Verify the tool resolves: `get_tools_for_agent(["my_new_tool"])` should
  return without error.
- For LLM-facing tools, test with a string input and verify JSON output.
- For deterministic functions, write unit tests covering edge cases.
- Run the relevant graph in paper mode to confirm end-to-end behavior.

### Checklist

- [ ] Function implemented with type hints and docstring
- [ ] `@tool` decorator applied (LLM-facing only)
- [ ] Registered in `TOOL_REGISTRY` (LLM-facing only)
- [ ] Added to agent YAML config (LLM-facing only)
- [ ] Error cases return structured JSON, not raw exceptions
- [ ] No side effects unless the tool name makes them obvious (e.g., `execute_*`)
- [ ] Tested in paper mode
