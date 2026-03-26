"""quantstack-ml — supervised ML training, inference, drift detection, ensembles."""
from quantstack.mcp.domains import Domain
from quantstack.mcp.server_factory import create_server, run_server

server = create_server(
    name="quantstack-ml",
    target=Domain.ML,
    instructions="Supervised ML: LightGBM/XGBoost/CatBoost training, SHAP analysis, concept drift detection, stacking ensembles, cross-sectional models.",
)

def main():
    run_server(server)

if __name__ == "__main__":
    main()
