"""
Paper Replications - Reusable implementations of key quant finance papers.

Modules:
- avellaneda_stoikov: Market making optimal quotes
- bouchaud_impact: Price impact and order flow models
- almgren_chriss: Optimal execution with market impact
- lob_features: ML features from limit order book
"""

from quantstack.core.research.paper_replications.almgren_chriss import (
    AlmgrenChrissExecutor,
    execution_cost,
    optimal_trajectory,
)
from quantstack.core.research.paper_replications.avellaneda_stoikov import (
    AvellanedaStoikovMM,
    optimal_spread,
    reservation_price,
)
from quantstack.core.research.paper_replications.bouchaud_impact import (
    BouchaudImpactModel,
    propagator_model,
)
from quantstack.core.research.paper_replications.lob_features import (
    LOBFeatureExtractor,
    order_imbalance,
)

__all__ = [
    "AvellanedaStoikovMM",
    "optimal_spread",
    "reservation_price",
    "BouchaudImpactModel",
    "propagator_model",
    "AlmgrenChrissExecutor",
    "optimal_trajectory",
    "execution_cost",
    "LOBFeatureExtractor",
    "order_imbalance",
]
