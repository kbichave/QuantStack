"""
Limit Order Book Feature Extraction.

References:
    Sirignano, J., & Cont, R. (2019).
    "Universal features of price formation in financial markets."

    Zhang, Z., Zohren, S., & Roberts, S. (2019).
    "DeepLOB: Deep convolutional neural networks for limit order books."

Key Features:
    - Order imbalance
    - Queue position and dynamics
    - Spread and mid-price features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class LOBSnapshot:
    """Snapshot of limit order book."""

    timestamp: float
    bid_prices: np.ndarray
    bid_sizes: np.ndarray
    ask_prices: np.ndarray
    ask_sizes: np.ndarray


def order_imbalance(
    bid_sizes: np.ndarray, ask_sizes: np.ndarray, levels: int = 5
) -> float:
    """
    Compute order imbalance at top N levels.

    OI = (Bid_Volume - Ask_Volume) / (Bid_Volume + Ask_Volume)

    Args:
        bid_sizes: Bid sizes
        ask_sizes: Ask sizes
        levels: Number of levels

    Returns:
        Order imbalance in [-1, 1]
    """
    bid_vol = np.sum(bid_sizes[:levels])
    ask_vol = np.sum(ask_sizes[:levels])

    total = bid_vol + ask_vol
    return (bid_vol - ask_vol) / total if total > 0 else 0.0


def queue_position_features(
    snapshot: LOBSnapshot, levels: int = 10
) -> Dict[str, float]:
    """Extract queue position features."""
    bid_prices = snapshot.bid_prices[:levels]
    bid_sizes = snapshot.bid_sizes[:levels]
    ask_prices = snapshot.ask_prices[:levels]
    ask_sizes = snapshot.ask_sizes[:levels]

    best_bid = bid_prices[0] if len(bid_prices) > 0 else 0
    best_ask = ask_prices[0] if len(ask_prices) > 0 else 0
    mid_price = (best_bid + best_ask) / 2
    spread = best_ask - best_bid

    return {
        "mid_price": mid_price,
        "spread": spread,
        "spread_bps": spread / mid_price * 10000 if mid_price > 0 else 0,
        "bid_depth_1": bid_sizes[0] if len(bid_sizes) > 0 else 0,
        "ask_depth_1": ask_sizes[0] if len(ask_sizes) > 0 else 0,
        "order_imbalance_1": order_imbalance(bid_sizes, ask_sizes, 1),
        "order_imbalance_5": order_imbalance(bid_sizes, ask_sizes, 5),
    }


class LOBFeatureExtractor:
    """
    Extract ML features from limit order book data.

    Example:
        extractor = LOBFeatureExtractor(n_levels=10)
        features = extractor.extract(snapshot)
    """

    def __init__(self, n_levels: int = 10, lookback: int = 50):
        self.n_levels = n_levels
        self.lookback = lookback
        self.history: List[LOBSnapshot] = []

    def extract(self, snapshot: LOBSnapshot) -> Dict[str, float]:
        """Extract all features from a LOB snapshot."""
        features = queue_position_features(snapshot, self.n_levels)

        for level in [1, 3, 5, 10]:
            if level <= self.n_levels:
                features[f"imbalance_{level}"] = order_imbalance(
                    snapshot.bid_sizes, snapshot.ask_sizes, level
                )

        self.history.append(snapshot)
        if len(self.history) > self.lookback:
            self.history.pop(0)

        if len(self.history) >= 5:
            features.update(self._compute_dynamics())

        return features

    def _compute_dynamics(self) -> Dict[str, float]:
        """Compute dynamic features from history."""
        features = {}

        mid_prices = [
            (s.bid_prices[0] + s.ask_prices[0]) / 2
            for s in self.history
            if len(s.bid_prices) > 0
        ]

        if len(mid_prices) >= 5:
            returns = np.diff(mid_prices) / np.array(mid_prices[:-1])
            features["return_5"] = np.sum(returns[-5:])
            features["volatility_5"] = np.std(returns[-5:])

        imbalances = [
            order_imbalance(s.bid_sizes, s.ask_sizes, 5) for s in self.history
        ]
        if len(imbalances) >= 5:
            features["imbalance_ma_5"] = np.mean(imbalances[-5:])

        return features

    def extract_batch(self, snapshots: List[LOBSnapshot]) -> pd.DataFrame:
        """Extract features for a batch of snapshots."""
        self.history = []
        all_features = []

        for snapshot in snapshots:
            features = self.extract(snapshot)
            features["timestamp"] = snapshot.timestamp
            all_features.append(features)

        return pd.DataFrame(all_features)
