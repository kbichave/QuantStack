"""quantstack-finrl — DRL training, evaluation, promotion, screening."""
from quantstack.mcp.domains import Domain
from quantstack.mcp.server_factory import create_server, run_server

server = create_server(
    name="quantstack-finrl",
    target=Domain.FINRL,
    instructions="Deep RL agents: PPO/A2C/SAC/TD3/DDPG/DQN training for execution timing, position sizing, alpha selection. Shadow mode, promotion gates.",
)

def main():
    run_server(server)

if __name__ == "__main__":
    main()
