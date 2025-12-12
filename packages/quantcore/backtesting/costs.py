"""
Production-grade transaction cost models for commodity trading.
"""


class ProductionCostModel:
    """
    Realistic transaction cost model for WTI/Brent spread trading.

    Based on CME CL and ICE BZ contract specifications:
    - CL: 1000 barrels per contract, $0.01 tick = $10
    - BZ: 1000 barrels per contract, $0.01 tick = $10
    """

    # Contract specs
    BARRELS_PER_CONTRACT = 1000
    TICK_SIZE = 0.01  # $0.01 per barrel
    TICK_VALUE = 10.0  # $10 per tick per contract

    # Cost components (per contract per side)
    COMMISSION = 2.50  # Broker commission per contract
    EXCHANGE_FEE = 1.50  # CME/ICE exchange fee
    CLEARING_FEE = 0.50  # Clearing fee

    # Spread costs
    BID_ASK_TICKS_CL = 1  # CL typically 1 tick spread
    BID_ASK_TICKS_BZ = 2  # BZ typically 2 ticks spread (less liquid)

    def __init__(
        self,
        slippage_model: str = "volatility",  # "fixed", "volatility", "size"
        base_slippage_ticks: float = 1.0,
        volatility_multiplier: float = 0.5,
        size_impact_factor: float = 0.1,  # Additional ticks per contract
    ):
        self.slippage_model = slippage_model
        self.base_slippage_ticks = base_slippage_ticks
        self.volatility_multiplier = volatility_multiplier
        self.size_impact_factor = size_impact_factor

    def calculate_slippage_ticks(
        self,
        n_contracts: int,
        volatility: float = 0.02,  # Daily volatility
        is_entry: bool = True,
    ) -> float:
        """Calculate slippage in ticks based on model."""
        if self.slippage_model == "fixed":
            return self.base_slippage_ticks

        elif self.slippage_model == "volatility":
            # Higher volatility = more slippage
            vol_factor = 1 + (volatility / 0.02 - 1) * self.volatility_multiplier
            return self.base_slippage_ticks * max(0.5, vol_factor)

        elif self.slippage_model == "size":
            # Larger size = more market impact
            size_penalty = (n_contracts - 1) * self.size_impact_factor
            return self.base_slippage_ticks + size_penalty

        return self.base_slippage_ticks

    def calculate_total_cost(
        self,
        n_contracts: int,
        volatility: float = 0.02,
        is_round_trip: bool = True,
    ) -> float:
        """
        Calculate total transaction cost in dollars.

        Returns total cost for opening AND closing the spread position.
        """
        sides = 2 if is_round_trip else 1
        legs = 2  # WTI + Brent

        # Fixed costs
        commission_total = self.COMMISSION * n_contracts * legs * sides
        exchange_total = self.EXCHANGE_FEE * n_contracts * legs * sides
        clearing_total = self.CLEARING_FEE * n_contracts * legs * sides

        # Bid-ask spread (crossing the spread)
        bid_ask_cost_cl = self.BID_ASK_TICKS_CL * self.TICK_VALUE * n_contracts * sides
        bid_ask_cost_bz = self.BID_ASK_TICKS_BZ * self.TICK_VALUE * n_contracts * sides

        # Slippage
        slippage_ticks = self.calculate_slippage_ticks(n_contracts, volatility)
        slippage_cost = slippage_ticks * self.TICK_VALUE * n_contracts * legs * sides

        total = (
            commission_total
            + exchange_total
            + clearing_total
            + bid_ask_cost_cl
            + bid_ask_cost_bz
            + slippage_cost
        )

        return total

    def cost_per_barrel(self, n_contracts: int, volatility: float = 0.02) -> float:
        """Convert total cost to cost per barrel for comparison."""
        total_cost = self.calculate_total_cost(n_contracts, volatility)
        total_barrels = n_contracts * self.BARRELS_PER_CONTRACT
        return total_cost / total_barrels
