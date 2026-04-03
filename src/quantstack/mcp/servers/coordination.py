"""quantstack-coord -- thin coordination MCP for orchestrator and agents.

Exposes only the ~8 hot-path tools needed during every loop iteration:
system status, portfolio state, regime, signal brief, heartbeat,
event bus (publish/poll), and trade execution.

All heavy computation (data, ML, backtesting, options, risk) is called
via Python imports in LangGraph node functions.
"""
from quantstack.mcp.domains import Domain
from quantstack.mcp.server_factory import create_server, run_server

server = create_server(
    name="quantstack-coord",
    target=Domain.COORDINATION,
    instructions=(
        "Coordination MCP: system status, portfolio state, regime, signal brief, "
        "heartbeat, event bus, trade execution. All other computation via Python imports."
    ),
)


def main():
    run_server(server)


if __name__ == "__main__":
    main()
