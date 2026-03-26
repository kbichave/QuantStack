"""quantstack-execution — trade execution, order management, alerts, coordination."""
from quantstack.mcp.domains import Domain
from quantstack.mcp.server_factory import create_server, run_server

server = create_server(
    name="quantstack-execution",
    target=Domain.EXECUTION,
    instructions="Trade execution, order management, position alerts, and loop coordination. The hot path for live trading.",
)

def main():
    run_server(server)

if __name__ == "__main__":
    main()
