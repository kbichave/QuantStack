# QuantPod Multi-Agent Architecture

QuantPod is a CrewAI-based multi-agent trading system designed for 24/7 market monitoring and intelligent trade execution.

## Overview

QuantPod orchestrates specialized AI agents that collaborate to:
- Monitor market conditions across multiple timeframes
- Analyze technical and fundamental data
- Manage risk and position sizing
- Execute trades with optimal timing

## Package Structure

```
packages/quant_pod/
├── crews/              # Agent crew definitions and assembly
│   ├── assembler.py    # Dynamic crew construction
│   ├── registry.py     # Agent and task registry
│   ├── schemas.py      # Pydantic schemas for crew config
│   ├── tools.py        # Crew-specific tools
│   └── trading_crew.py # Main trading crew implementation
├── flows/              # Orchestrated workflows
│   └── trading_day_flow.py  # Daily trading flow
├── knowledge/          # Persistent knowledge storage
│   ├── models.py       # Knowledge data models
│   ├── policy_store.py # Trading policy storage
│   └── store.py        # Knowledge store implementation
├── memory/             # Agent communication
│   ├── blackboard.py   # Blackboard pattern implementation
│   └── mem0_client.py  # Mem0 integration
├── prompts/            # Agent prompts and instructions
│   ├── assistant/      # Trading assistant prompts
│   ├── ics/            # Individual contributor prompts
│   ├── pod_managers/   # Pod manager prompts
│   └── supertrader/    # Super trader prompts
└── tools/              # MCP bridge and utilities
    ├── alphavantage_tools.py
    ├── knowledge_tools.py
    ├── mcp_bridge.py   # Bridge to QuantCore MCP
    └── memory_tools.py
```

## Agent Hierarchy

QuantPod uses a hierarchical agent structure:

```
                    ┌─────────────────┐
                    │  Super Trader   │
                    │   (Orchestrator)│
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Market Monitor│   │   Quant Pod   │   │   Risk Pod    │
│  Pod Manager  │   │    Manager    │   │    Manager    │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  ICs (Agents) │   │  ICs (Agents) │   │  ICs (Agents) │
│ - Regime      │   │ - StatArb     │   │ - Risk Limits │
│ - Snapshot    │   │ - Options Vol │   │ - Calendar    │
└───────────────┘   └───────────────┘   └───────────────┘
```

### Agent Types

**Super Trader**
- Top-level orchestrator
- Makes final trading decisions
- Synthesizes inputs from all pods

**Pod Managers**
- Coordinate specialized agent groups
- Aggregate IC outputs
- Report to Super Trader

**Individual Contributors (ICs)**
- Specialized analysis agents
- Focus on specific signals/metrics
- Report to Pod Managers

## Pods and Their ICs

### Market Monitor Pod
Tracks market conditions and regime changes.

| IC Agent | Responsibility |
|----------|---------------|
| Market Snapshot IC | Current prices, breadth, sentiment |
| Regime Detector IC | Bull/bear/sideways classification |

### Technicals Pod
Technical analysis across timeframes.

| IC Agent | Responsibility |
|----------|---------------|
| Structure Levels IC | Support/resistance identification |
| Trend Momentum IC | Trend strength and direction |
| Volatility IC | Vol regime and expansion/contraction |

### Quant Pod
Quantitative signal generation.

| IC Agent | Responsibility |
|----------|---------------|
| StatArb IC | Statistical arbitrage signals |
| Options Vol IC | Volatility surface analysis |

### Risk Pod
Risk management and controls.

| IC Agent | Responsibility |
|----------|---------------|
| Risk Limits IC | Position and exposure limits |
| Calendar Events IC | Earnings, FOMC, economic releases |

### Data Pod
Data ingestion and quality.

| IC Agent | Responsibility |
|----------|---------------|
| Data Ingestion IC | Market data fetching and validation |

## Crews

Crews are task-oriented groupings of agents:

```python
from quant_pod.crews import TradingCrew

# Create a trading crew
crew = TradingCrew(
    llm_model="gpt-4",
    verbose=True
)

# Execute daily analysis
result = crew.kickoff(
    inputs={
        "symbols": ["SPY", "QQQ"],
        "date": "2024-01-15"
    }
)
```

### Crew Assembly

The `CrewAssembler` dynamically constructs crews based on configuration:

```python
from quant_pod.crews import CrewAssembler

assembler = CrewAssembler()
crew = assembler.build(
    agents=["market_snapshot", "regime_detector", "risk_limits"],
    tasks=["morning_scan", "risk_check"]
)
```

## Flows

Flows orchestrate multi-step trading workflows:

```python
from quant_pod.flows import TradingDayFlow

flow = TradingDayFlow()

# Run the complete trading day workflow
await flow.run(
    pre_market=True,
    intraday_interval=15,  # minutes
    post_market=True
)
```

### Trading Day Flow

```
Pre-Market (6:00 AM)
    │
    ▼
┌─────────────────┐
│ Market Snapshot │ ──► Overnight gaps, futures, global markets
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Calendar Check  │ ──► Earnings, economic releases
└────────┬────────┘
         │
         ▼
Market Open (9:30 AM)
         │
         ▼
┌─────────────────┐
│ Intraday Loop   │ ◄──┐
│ (every 15 min)  │    │
│ - Tech Analysis │    │
│ - Risk Monitor  │    │
│ - Signal Gen    │ ───┘
└────────┬────────┘
         │
         ▼
Market Close (4:00 PM)
         │
         ▼
┌─────────────────┐
│ Post-Market     │ ──► P&L reconciliation, journal
└─────────────────┘
```

## Knowledge Store

Persistent storage for trading insights and learned policies:

```python
from quant_pod.knowledge import KnowledgeStore, Policy

store = KnowledgeStore(db_path="knowledge.db")

# Store a trading insight
store.add_insight(
    symbol="AAPL",
    insight_type="support_level",
    value=175.50,
    confidence=0.85
)

# Store a learned policy
policy = Policy(
    name="earnings_avoidance",
    rule="Reduce position size 50% 2 days before earnings",
    effectiveness=0.72
)
store.add_policy(policy)
```

## Memory (Blackboard Pattern)

Agents communicate via a shared blackboard:

```python
from quant_pod.memory import Blackboard

board = Blackboard()

# Agent writes to blackboard
board.write(
    key="regime",
    value="bullish",
    source="regime_detector_ic",
    timestamp=datetime.now()
)

# Another agent reads
regime = board.read("regime")
```

### Memory Integration (Mem0)

Long-term memory via Mem0 for persistent agent learning:

```python
from quant_pod.memory import Mem0Client

mem = Mem0Client()

# Store interaction memory
mem.add(
    messages=[
        {"role": "user", "content": "What's the current SPY regime?"},
        {"role": "assistant", "content": "Bullish with momentum divergence"}
    ],
    user_id="trading_system"
)

# Retrieve relevant memories
memories = mem.search("SPY regime analysis", limit=5)
```

## Tools

### MCP Bridge

Connects agents to QuantCore capabilities via MCP:

```python
from quant_pod.tools import MCPBridge

bridge = MCPBridge(server_url="http://localhost:8080")

# Call QuantCore indicators
rsi = await bridge.call(
    tool="compute_indicator",
    params={"symbol": "SPY", "indicator": "RSI", "period": 14}
)
```

### Alpha Vantage Tools

Direct market data access:

```python
from quant_pod.tools import AlphaVantageTools

av = AlphaVantageTools(api_key="your_key")

# Get real-time quote
quote = av.get_quote("AAPL")

# Get news sentiment
sentiment = av.get_news_sentiment("AAPL")
```

## Prompt Engineering

Agent prompts are stored as paired JSON + Markdown files:

```
prompts/
├── ics/
│   └── market_monitor/
│       ├── regime_detector_ic.json   # Structured config
│       └── regime_detector_ic.md     # Detailed instructions
```

**JSON Config Example:**
```json
{
  "name": "Regime Detector IC",
  "role": "Market regime classification specialist",
  "goal": "Identify current market regime (bull/bear/sideways)",
  "backstory": "Expert in regime switching models..."
}
```

**Markdown Instructions:**
```markdown
# Regime Detector IC

## Objective
Classify the current market regime using multiple indicators.

## Methodology
1. Analyze trend direction (200 EMA)
2. Measure momentum (RSI, MACD)
3. Assess volatility regime (VIX levels)
...
```

## Configuration

QuantPod now ships its runtime configuration inside the package:

- Agent prompts live under `packages/quant_pod/prompts/`
- Task definitions live in `packages/quant_pod/crews/config/tasks.yaml`
- Schedules and trading parameters are encoded in the crews/flows code (no external YAML required)

To customize, copy those files into your own config path and point your runner/CLI to the overrides.

## Integration with QuantCore

QuantPod leverages QuantCore for quantitative analysis:

```python
# Agent using QuantCore via MCP
class TechAnalysisIC(Agent):
    def analyze(self, symbol: str):
        # Get indicators from QuantCore
        indicators = self.mcp_bridge.call(
            "compute_all_indicators",
            {"symbol": symbol, "timeframe": "4h"}
        )
        
        # Get regime from QuantCore
        regime = self.mcp_bridge.call(
            "detect_regime",
            {"returns": indicators["returns"]}
        )
        
        return self.synthesize(indicators, regime)
```

## Deployment

```bash
# Start QuantPod daemon
quant-pod-daemon  # uses packaged defaults unless you pass --config

# Run single analysis
quant-pod analyze --symbols SPY,QQQ --verbose
```
