"""quantstack-research — backtesting, walk-forward, strategy lifecycle, statistical validation."""
from quantstack.mcp.domains import Domain
from quantstack.mcp.server_factory import create_server

server = create_server(
    name="quantstack-research",
    target=Domain.RESEARCH,
    instructions="Strategy research: backtesting (single/MTF/options), walk-forward validation, IC, PBO, alpha decay, strategy CRUD, promotion.",
)

def main():
    server.run()

if __name__ == "__main__":
    main()
