"""quantstack-risk — VaR, stress testing, drawdown analysis, risk metrics."""
from quantstack.mcp.domains import Domain
from quantstack.mcp.server_factory import create_server

server = create_server(
    name="quantstack-risk",
    target=Domain.RISK,
    instructions="Risk analytics: VaR/CVaR, Monte Carlo stress testing, max drawdown, Sortino/Calmar ratios, position sizing validation.",
)

def main():
    server.run()

if __name__ == "__main__":
    main()
