"""quantstack-intel — capitulation, institutional accumulation, macro signals, NLP."""
from quantstack.mcp.domains import Domain
from quantstack.mcp.server_factory import create_server, run_server

server = create_server(
    name="quantstack-intel",
    target=Domain.INTEL,
    instructions="Market intelligence: capitulation scoring (tier_3), institutional accumulation (tier_3), credit/macro signals (tier_4), market breadth, NLP sentiment, cross-domain intel.",
)

def main():
    run_server(server)

if __name__ == "__main__":
    main()
