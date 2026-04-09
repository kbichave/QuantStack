# Section 07: Deep Hedging Network

## Objective

Implement a deep hedging network (Buehler et al. 2019) that learns optimal hedge ratios by minimizing CVaR of hedging error, accounting for transaction costs. This replaces Black-Scholes delta hedging when the deep hedge proves superior (>10% CVaR improvement).

**Prerequisite**: P08 (options market-making infrastructure) must be completed before this section can be deployed. Build the module now; enable via feature flag later.

## Files to Create/Modify

### New Files

- **`src/quantstack/core/options/deep_hedging.py`** — Deep hedging LSTM network, training, and inference.

### Modified Files

- **`src/quantstack/config/feedback_flags.py`** — Add `deep_hedging_enabled()` feature flag (default `False`).
- **`src/quantstack/core/options/engine.py`** — Add conditional path: when `deep_hedging_enabled()`, use deep hedge ratio instead of BS delta.

## Implementation Details

### `src/quantstack/core/options/deep_hedging.py`

```
class DeepHedgingNetwork(nn.Module):
    """LSTM + FC network that learns hedge ratios from market state.
    
    Architecture (Buehler et al. 2019):
    - Input: (portfolio_greeks, underlying_price, time_to_expiry, iv, moneyness)
    - LSTM: 2 layers, hidden_size=64
    - FC head: 64 -> 32 -> 1 (hedge ratio = shares of underlying to hold)
    - Output: single float — hedge ratio in [-2.0, 2.0] range (clipped)
    """

    def __init__(
        self,
        input_dim: int = 8,        # greeks(4) + price + tte + iv + moneyness
        hidden_dim: int = 64,
        n_lstm_layers: int = 2,
        dropout: float = 0.1,
    ):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map market state sequence to hedge ratio."""
```

```
class DeepHedgingTrainer:
    """Train deep hedging network on simulated paths."""

    def __init__(
        self,
        transaction_cost_bps: float = 10.0,  # 0.1% = 10 bps per rebalance
        rebalance_freq: str = "daily",
        cvar_alpha: float = 0.05,             # CVaR at 5th percentile
        n_paths: int = 10_000,
        n_steps: int = 63,                    # ~3 months of daily rebalancing
    ):
        ...

    def simulate_paths(
        self,
        S0: float,
        sigma: float,
        r: float = 0.05,
        kappa: float = 2.0,     # mean reversion speed (stochastic vol)
        theta: float = 0.04,    # long-run variance
        xi: float = 0.3,        # vol of vol
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate price paths using GBM with stochastic volatility (Heston).
        
        Returns (price_paths, vol_paths) each of shape (n_paths, n_steps).
        """

    def compute_hedging_error(
        self,
        hedge_ratios: torch.Tensor,    # (n_paths, n_steps)
        price_paths: torch.Tensor,     # (n_paths, n_steps)
        option_payoffs: torch.Tensor,  # (n_paths,)
    ) -> torch.Tensor:
        """Compute P&L of hedging strategy minus option payoff.
        
        Includes transaction costs for each rebalance.
        Returns hedging errors of shape (n_paths,).
        """

    def cvar_loss(self, hedging_errors: torch.Tensor) -> torch.Tensor:
        """CVaR loss — expected shortfall at alpha percentile.
        
        This is the training objective: minimize the tail risk of hedging errors.
        """

    def train(self, model: DeepHedgingNetwork, n_epochs: int = 200) -> DeepHedgeTrainResult:
        """Full training loop.
        
        Steps:
        1. Simulate n_paths price paths with stochastic vol
        2. For each path, run model to get hedge ratios at each step
        3. Compute hedging error (hedge P&L - option payoff - transaction costs)
        4. Minimize CVaR of hedging error
        5. Compare against BS delta hedge on same paths
        
        Returns result with CVaR comparison vs BS delta hedge.
        """

    def benchmark_vs_bs(
        self,
        model: DeepHedgingNetwork,
        price_paths: torch.Tensor,
    ) -> dict:
        """Run both deep hedge and BS delta hedge on same paths.
        
        Returns:
            deep_cvar: float
            bs_cvar: float
            improvement_pct: float — (bs_cvar - deep_cvar) / bs_cvar * 100
            deploy_recommended: bool — True if improvement > 10%
        """
```

```
@dataclass
class DeepHedgeTrainResult:
    deep_cvar: float
    bs_cvar: float
    improvement_pct: float
    deploy_recommended: bool
    n_paths: int
    n_epochs: int
    checkpoint_path: str
```

### Feature Flag

In `feedback_flags.py`, add:
```python
def deep_hedging_enabled() -> bool:
    """Deep hedging replaces BS delta when True. Requires P08."""
    return _flag("deep_hedging_enabled", default=False)
```

### Engine Integration

In `engine.py`, add conditional logic:
```python
if deep_hedging_enabled():
    hedge_ratio = deep_hedging_model.predict(market_state)
else:
    hedge_ratio = bs_delta  # existing behavior
```

The deep hedging model runs alongside BS delta during A/B testing. Both hedge ratios are logged; only the active one is executed.

## Dependencies

- **PyPI**: `torch` (PyTorch — already needed for GNN)
- **Internal**: `quantstack.core.options.pricing` (BS delta for benchmark), `quantstack.config.feedback_flags`
- **Cross-plan**: P08 (options market-making) must be complete before `deep_hedging_enabled()` is set to True

## Test Requirements

### `tests/unit/options/test_deep_hedging.py`

1. **CVaR vs BS**: On simulated GBM paths, verify deep hedge CVaR <= BS CVaR after training (may need 50+ epochs).
2. **Transaction cost accounting**: With high transaction costs (50 bps), verify deep hedge learns to rebalance less frequently.
3. **Hedge ratio bounds**: Output always in [-2.0, 2.0] range.
4. **Feature flag off**: When `deep_hedging_enabled()` returns False, engine uses BS delta.
5. **Stochastic vol paths**: Verify simulated paths have realistic properties (positive prices, vol clustering).
6. **Benchmark comparison**: `benchmark_vs_bs` returns correct structure with improvement_pct.

## Acceptance Criteria

- [ ] `DeepHedgingNetwork` produces hedge ratios from market state inputs
- [ ] Training minimizes CVaR of hedging error with transaction costs
- [ ] Stochastic volatility path simulation produces realistic price paths
- [ ] Benchmark comparison against BS delta hedge works correctly
- [ ] Feature flag `deep_hedging_enabled()` defaults to False
- [ ] Engine integration is gated behind feature flag
- [ ] Deploy recommendation requires >10% CVaR improvement
- [ ] All unit tests pass
- [ ] No GPU required for inference
