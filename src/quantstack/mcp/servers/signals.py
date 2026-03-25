"""quantstack-signals — signal briefs, regime classification, intraday status."""
from quantstack.mcp.domains import Domain
from quantstack.mcp.server_factory import create_server

server = create_server(
    name="quantstack-signals",
    target=Domain.SIGNALS,
    instructions="Signal analysis (15 collectors, 2-6s), regime classification, intraday status, TCA reports.",
)

def main():
    server.run()

if __name__ == "__main__":
    main()
