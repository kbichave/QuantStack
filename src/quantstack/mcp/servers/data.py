"""quantstack-data — OHLCV, technical indicators, fundamentals, microstructure."""
from quantstack.mcp.domains import Domain
from quantstack.mcp.server_factory import create_server

server = create_server(
    name="quantstack-data",
    target=Domain.DATA,
    instructions="Market data: OHLCV loading, 200+ technical indicators, fundamentals, microstructure analysis, Alpha Vantage queries.",
)

def main():
    server.run()

if __name__ == "__main__":
    main()
