"""quantstack-options — Greeks, IV surface, vol forecasting, option pricing."""
from quantstack.mcp.domains import Domain
from quantstack.mcp.server_factory import create_server

server = create_server(
    name="quantstack-options",
    target=Domain.OPTIONS,
    instructions="Options analytics: Greeks computation, IV surface, GARCH/EGARCH vol forecasting, option pricing, skew analysis, term structure.",
)

def main():
    server.run()

if __name__ == "__main__":
    main()
