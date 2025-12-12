"""
Kalman Filter for State-Space Models.

Linear Gaussian state estimation and smoothing.
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class KalmanState:
    """State of Kalman filter."""

    x: np.ndarray  # State estimate
    P: np.ndarray  # State covariance
    innovation: float  # Measurement residual
    log_likelihood: float


class KalmanFilter:
    """
    Linear Kalman Filter.

    State-space model:
        x_t = F @ x_{t-1} + w_t,  w_t ~ N(0, Q)
        y_t = H @ x_t + v_t,      v_t ~ N(0, R)

    Example:
        kf = KalmanFilter(
            F=np.array([[1]]),
            H=np.array([[1]]),
            Q=np.array([[0.01]]),
            R=np.array([[0.1]]),
        )
        filtered = kf.filter(observations)
    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray = None,
        P0: np.ndarray = None,
    ):
        self.F = np.atleast_2d(F)
        self.H = np.atleast_2d(H)
        self.Q = np.atleast_2d(Q)
        self.R = np.atleast_2d(R)

        self.n = self.F.shape[0]
        self.x0 = x0 if x0 is not None else np.zeros(self.n)
        self.P0 = P0 if P0 is not None else np.eye(self.n) * 10

    def predict(self, x: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction step."""
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(self, x_pred: np.ndarray, P_pred: np.ndarray, y: float) -> KalmanState:
        """Update step with new observation."""
        y_pred = self.H @ x_pred
        innovation = y - y_pred[0]

        S = self.H @ P_pred @ self.H.T + self.R
        S_scalar = S[0, 0]

        K = P_pred @ self.H.T / S_scalar

        x = x_pred + K.flatten() * innovation
        P = (np.eye(self.n) - np.outer(K, self.H)) @ P_pred

        log_lik = -0.5 * (np.log(2 * np.pi * S_scalar) + innovation**2 / S_scalar)

        return KalmanState(x=x, P=P, innovation=innovation, log_likelihood=log_lik)

    def filter(self, observations: np.ndarray) -> List[KalmanState]:
        """Run Kalman filter on observations."""
        T = len(observations)
        states = []

        x = self.x0.copy()
        P = self.P0.copy()

        for t in range(T):
            x_pred, P_pred = self.predict(x, P)

            if not np.isnan(observations[t]):
                state = self.update(x_pred, P_pred, observations[t])
                x = state.x
                P = state.P
            else:
                state = KalmanState(x=x_pred, P=P_pred, innovation=0, log_likelihood=0)
                x = x_pred
                P = P_pred

            states.append(state)

        return states

    def smooth(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """RTS smoother (backward pass after forward filtering)."""
        filtered = self.filter(observations)
        T = len(observations)

        x_smooth = np.zeros((T, self.n))
        P_smooth = np.zeros((T, self.n, self.n))

        x_smooth[-1] = filtered[-1].x
        P_smooth[-1] = filtered[-1].P

        for t in range(T - 2, -1, -1):
            x_pred, P_pred = self.predict(filtered[t].x, filtered[t].P)
            J = filtered[t].P @ self.F.T @ np.linalg.inv(P_pred)

            x_smooth[t] = filtered[t].x + J @ (x_smooth[t + 1] - x_pred)
            P_smooth[t] = filtered[t].P + J @ (P_smooth[t + 1] - P_pred) @ J.T

        return x_smooth, P_smooth


class LocalLevelModel(KalmanFilter):
    """Local Level Model (Random Walk + Noise)."""

    def __init__(self, sigma_eta: float = 0.1, sigma_epsilon: float = 1.0):
        super().__init__(
            F=np.array([[1]]),
            H=np.array([[1]]),
            Q=np.array([[sigma_eta**2]]),
            R=np.array([[sigma_epsilon**2]]),
        )
