"""
Execution Algorithms.

TWAP, VWAP, IS minimization, and POV algorithms.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from quantcore.microstructure.order_book import Side


@dataclass
class ExecutionSlice:
    """Single slice of parent order."""

    time: float
    quantity: float


@dataclass
class ExecutionPlan:
    """Execution plan for parent order."""

    parent_size: float
    parent_side: Side
    horizon: float
    slices: List[ExecutionSlice]
    algo_name: str


class ExecutionAlgo(ABC):
    """Base class for execution algorithms."""

    @abstractmethod
    def create_plan(
        self, order_size: float, side: Side, horizon: float
    ) -> ExecutionPlan:
        pass


class TWAPExecutor(ExecutionAlgo):
    """
    Time-Weighted Average Price execution.

    Splits order evenly over time.
    """

    def __init__(self, n_slices: int = 20):
        self.n_slices = n_slices

    def create_plan(
        self, order_size: float, side: Side, horizon: float
    ) -> ExecutionPlan:
        slice_size = order_size / self.n_slices
        slice_interval = horizon / self.n_slices

        slices = [
            ExecutionSlice(time=i * slice_interval, quantity=slice_size)
            for i in range(self.n_slices)
        ]

        return ExecutionPlan(
            parent_size=order_size,
            parent_side=side,
            horizon=horizon,
            slices=slices,
            algo_name="TWAP",
        )


class VWAPExecutor(ExecutionAlgo):
    """
    Volume-Weighted Average Price execution.

    Trades proportional to expected volume profile.
    """

    def __init__(self, volume_profile: Optional[np.ndarray] = None, n_slices: int = 20):
        self.n_slices = n_slices

        if volume_profile is not None:
            self.volume_profile = volume_profile / volume_profile.sum()
        else:
            # Default U-shaped profile
            x = np.linspace(0, 1, n_slices)
            profile = 1 + 0.5 * (np.cos(np.pi * x) ** 2)
            self.volume_profile = profile / profile.sum()

    def create_plan(
        self, order_size: float, side: Side, horizon: float
    ) -> ExecutionPlan:
        slice_interval = horizon / self.n_slices

        slices = [
            ExecutionSlice(
                time=i * slice_interval, quantity=order_size * self.volume_profile[i]
            )
            for i in range(self.n_slices)
        ]

        return ExecutionPlan(
            parent_size=order_size,
            parent_side=side,
            horizon=horizon,
            slices=slices,
            algo_name="VWAP",
        )


class ISExecutor(ExecutionAlgo):
    """
    Implementation Shortfall minimization executor.

    Almgren-Chriss optimal trajectory.
    """

    def __init__(
        self,
        volatility: float,
        daily_volume: float,
        eta: float = 2.5e-7,
        risk_aversion: float = 1e-6,
        n_slices: int = 20,
    ):
        self.volatility = volatility
        self.daily_volume = daily_volume
        self.eta = eta
        self.risk_aversion = risk_aversion
        self.n_slices = n_slices

    def create_plan(
        self, order_size: float, side: Side, horizon: float
    ) -> ExecutionPlan:
        kappa = np.sqrt(self.risk_aversion * self.volatility**2 / (self.eta + 1e-10))

        times = np.linspace(0, horizon, self.n_slices + 1)

        if kappa * horizon < 0.001:
            remaining = order_size * (1 - times / horizon)
        else:
            sinh_kappa_T = np.sinh(kappa * horizon)
            remaining = order_size * np.sinh(kappa * (horizon - times)) / sinh_kappa_T

        trade_sizes = -np.diff(remaining)

        slices = [
            ExecutionSlice(time=times[i], quantity=trade_sizes[i])
            for i in range(self.n_slices)
        ]

        return ExecutionPlan(
            parent_size=order_size,
            parent_side=side,
            horizon=horizon,
            slices=slices,
            algo_name="IS",
        )
