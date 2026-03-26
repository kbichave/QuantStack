"""quantstack-portfolio — portfolio state, optimization, attribution, feedback."""
from quantstack.mcp.domains import Domain
from quantstack.mcp.server_factory import create_server, run_server

server = create_server(
    name="quantstack-portfolio",
    target=Domain.PORTFOLIO,
    instructions="Portfolio state, HRP/MVO optimization, P&L attribution, and fill quality feedback.",
)

def main():
    run_server(server)

if __name__ == "__main__":
    main()
