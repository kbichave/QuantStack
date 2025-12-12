"""
Hidden Markov Model for regime detection.

Uses hmmlearn for statistical regime classification.
"""

from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger

try:
    from hmmlearn import hmm

    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not available. Install with: pip install hmmlearn")


class HMMRegimeState(Enum):
    """HMM regime states."""

    LOW_VOL_BULL = 0
    HIGH_VOL_BULL = 1
    LOW_VOL_BEAR = 2
    HIGH_VOL_BEAR = 3


@dataclass
class HMMRegimeResult:
    """Result from HMM regime detection."""

    state: HMMRegimeState
    state_probabilities: Dict[HMMRegimeState, float]
    transition_matrix: np.ndarray
    expected_duration: float
    regime_stability: float  # How stable is current regime

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "state": self.state.name,
            "state_value": self.state.value,
            "probabilities": {s.name: p for s, p in self.state_probabilities.items()},
            "expected_duration": self.expected_duration,
            "regime_stability": self.regime_stability,
        }


class HMMRegimeModel:
    """
    Hidden Markov Model for regime detection.

    Uses a 4-state Gaussian HMM to classify market regimes:
    - State 0: Low volatility bull (trending up, low vol)
    - State 1: High volatility bull (trending up, high vol)
    - State 2: Low volatility bear (trending down, low vol)
    - State 3: High volatility bear (trending down, high vol)

    Features used:
    - Returns
    - Volatility
    - Volume changes
    """

    def __init__(
        self,
        n_states: int = 4,
        lookback: int = 252,
        min_train_samples: int = 100,
        n_iter: int = 100,
        random_state: int = 42,
    ):
        """
        Initialize HMM regime model.

        Args:
            n_states: Number of hidden states (default 4)
            lookback: Lookback period for training
            min_train_samples: Minimum samples for training
            n_iter: Max iterations for EM algorithm
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.lookback = lookback
        self.min_train_samples = min_train_samples
        self.n_iter = n_iter
        self.random_state = random_state

        self.model: Optional[hmm.GaussianHMM] = None
        self.is_fitted = False
        self._state_mapping: Dict[int, HMMRegimeState] = {}

    def fit(self, df: pd.DataFrame) -> "HMMRegimeModel":
        """
        Fit HMM model to data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Self for chaining
        """
        if not HMM_AVAILABLE:
            logger.warning("hmmlearn not available, using fallback")
            return self

        if len(df) < self.min_train_samples:
            logger.warning(
                f"Insufficient data for HMM training: {len(df)} < {self.min_train_samples}"
            )
            return self

        # Prepare features
        features = self._prepare_features(df)

        if features is None or len(features) < self.min_train_samples:
            logger.warning("Failed to prepare features for HMM")
            return self

        try:
            # Initialize and fit HMM
            self.model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=self.random_state,
            )

            self.model.fit(features)
            self.is_fitted = True

            # Map states to interpretable regimes
            self._map_states(df, features)

            logger.info(f"HMM model fitted with {self.n_states} states")

        except Exception as e:
            logger.error(f"Failed to fit HMM model: {e}")
            self.is_fitted = False

        return self

    def predict(self, df: pd.DataFrame) -> HMMRegimeResult:
        """
        Predict current regime.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            HMMRegimeResult with regime information
        """
        # Fallback if not fitted or hmmlearn not available
        if not self.is_fitted or self.model is None:
            return self._fallback_predict(df)

        # Prepare features
        features = self._prepare_features(df)

        if features is None or len(features) == 0:
            return self._fallback_predict(df)

        try:
            # Get state probabilities
            log_prob, state_sequence = self.model.decode(features, algorithm="viterbi")
            posteriors = self.model.predict_proba(features)

            # Current state
            current_state_idx = state_sequence[-1]
            current_state = self._state_mapping.get(
                current_state_idx, HMMRegimeState(current_state_idx % 4)
            )

            # State probabilities
            current_probs = posteriors[-1]
            state_probs = {
                self._state_mapping.get(i, HMMRegimeState(i % 4)): float(
                    current_probs[i]
                )
                for i in range(self.n_states)
            }

            # Expected duration (from transition matrix)
            trans_mat = self.model.transmat_
            expected_duration = 1.0 / (
                1.0 - trans_mat[current_state_idx, current_state_idx] + 1e-10
            )

            # Regime stability (how confident we are in current state)
            regime_stability = float(current_probs[current_state_idx])

            return HMMRegimeResult(
                state=current_state,
                state_probabilities=state_probs,
                transition_matrix=trans_mat,
                expected_duration=expected_duration,
                regime_stability=regime_stability,
            )

        except Exception as e:
            logger.error(f"HMM prediction failed: {e}")
            return self._fallback_predict(df)

    def predict_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict regime for entire series.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series of regime states
        """
        if not self.is_fitted or self.model is None:
            return self._fallback_series(df)

        features = self._prepare_features(df)

        if features is None:
            return self._fallback_series(df)

        try:
            state_sequence = self.model.predict(features)

            # Map to regime names
            regime_names = [
                self._state_mapping.get(s, HMMRegimeState(s % 4)).name
                for s in state_sequence
            ]

            # Create series with proper index
            idx = df.index[len(df) - len(regime_names) :]
            return pd.Series(regime_names, index=idx, name="hmm_regime")

        except Exception as e:
            logger.error(f"HMM series prediction failed: {e}")
            return self._fallback_series(df)

    def _prepare_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for HMM."""
        if len(df) < 5:
            return None

        # Calculate features
        returns = df["close"].pct_change()
        volatility = returns.rolling(20).std()

        # Volume change (if volume available)
        if "volume" in df.columns:
            vol_change = df["volume"].pct_change()
        else:
            vol_change = pd.Series(0, index=df.index)

        # Momentum
        momentum = df["close"].pct_change(5)

        # Combine features
        feature_df = pd.DataFrame(
            {
                "returns": returns,
                "volatility": volatility,
                "vol_change": vol_change,
                "momentum": momentum,
            }
        ).dropna()

        if len(feature_df) < 10:
            return None

        # Standardize
        features = feature_df.values
        mean = np.nanmean(features, axis=0)
        std = np.nanstd(features, axis=0)
        std[std == 0] = 1

        features = (features - mean) / std

        # Handle any remaining NaN/inf
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

        return features

    def _map_states(self, df: pd.DataFrame, features: np.ndarray) -> None:
        """Map HMM states to interpretable regimes."""
        if self.model is None:
            return

        try:
            states = self.model.predict(features)
            returns = df["close"].pct_change().dropna().values[-len(states) :]
            volatility = (
                df["close"]
                .pct_change()
                .rolling(20)
                .std()
                .dropna()
                .values[-len(states) :]
            )

            # Calculate average return and vol for each state
            state_stats = {}
            for s in range(self.n_states):
                mask = states == s
                if mask.sum() > 0:
                    avg_ret = np.nanmean(returns[mask])
                    avg_vol = np.nanmean(volatility[mask])
                    state_stats[s] = {"return": avg_ret, "vol": avg_vol}

            # Determine median volatility
            all_vols = [
                v["vol"] for v in state_stats.values() if not np.isnan(v["vol"])
            ]
            med_vol = np.median(all_vols) if all_vols else 0

            # Map states
            for s, stats in state_stats.items():
                is_bull = stats["return"] > 0
                is_high_vol = stats["vol"] > med_vol

                if is_bull and not is_high_vol:
                    self._state_mapping[s] = HMMRegimeState.LOW_VOL_BULL
                elif is_bull and is_high_vol:
                    self._state_mapping[s] = HMMRegimeState.HIGH_VOL_BULL
                elif not is_bull and not is_high_vol:
                    self._state_mapping[s] = HMMRegimeState.LOW_VOL_BEAR
                else:
                    self._state_mapping[s] = HMMRegimeState.HIGH_VOL_BEAR

        except Exception as e:
            logger.warning(f"Failed to map HMM states: {e}")
            # Default mapping
            for i in range(self.n_states):
                self._state_mapping[i] = HMMRegimeState(i % 4)

    def _fallback_predict(self, df: pd.DataFrame) -> HMMRegimeResult:
        """Fallback prediction when HMM is not available."""
        # Simple rule-based regime detection
        returns = df["close"].pct_change()
        avg_ret = returns.tail(20).mean()
        vol = returns.tail(20).std()
        avg_vol = returns.rolling(60).std().iloc[-1] if len(df) > 60 else vol

        is_bull = avg_ret > 0
        is_high_vol = vol > avg_vol

        if is_bull and not is_high_vol:
            state = HMMRegimeState.LOW_VOL_BULL
        elif is_bull and is_high_vol:
            state = HMMRegimeState.HIGH_VOL_BULL
        elif not is_bull and not is_high_vol:
            state = HMMRegimeState.LOW_VOL_BEAR
        else:
            state = HMMRegimeState.HIGH_VOL_BEAR

        # Equal probabilities for fallback
        probs = {s: 0.25 for s in HMMRegimeState}
        probs[state] = 0.7

        return HMMRegimeResult(
            state=state,
            state_probabilities=probs,
            transition_matrix=np.eye(4),
            expected_duration=10,
            regime_stability=0.5,
        )

    def _fallback_series(self, df: pd.DataFrame) -> pd.Series:
        """Fallback series prediction."""
        # Rolling regime detection
        window = 20
        regimes = []

        for i in range(len(df)):
            if i < window:
                regimes.append("LOW_VOL_BULL")
                continue

            subset = df.iloc[max(0, i - window) : i + 1]
            returns = subset["close"].pct_change()
            avg_ret = returns.mean()
            vol = returns.std()

            is_bull = avg_ret > 0
            is_high_vol = (
                vol > returns.rolling(60).std().iloc[-1] if len(df) > 60 else False
            )

            if is_bull and not is_high_vol:
                regimes.append("LOW_VOL_BULL")
            elif is_bull and is_high_vol:
                regimes.append("HIGH_VOL_BULL")
            elif not is_bull and not is_high_vol:
                regimes.append("LOW_VOL_BEAR")
            else:
                regimes.append("HIGH_VOL_BEAR")

        return pd.Series(regimes, index=df.index, name="hmm_regime")
