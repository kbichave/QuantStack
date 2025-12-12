"""
Particle Filter (Sequential Monte Carlo).

Nonlinear filtering for state-space models.
"""

import numpy as np
from typing import Callable, Optional, List
from dataclasses import dataclass


@dataclass
class ParticleState:
    """State of particle filter."""

    particles: np.ndarray
    weights: np.ndarray
    estimate: float
    variance: float
    ess: float  # Effective sample size


def systematic_resample(weights: np.ndarray) -> np.ndarray:
    """Systematic resampling."""
    n = len(weights)
    cumsum = np.cumsum(weights)
    u = (np.arange(n) + np.random.uniform()) / n
    indices = np.searchsorted(cumsum, u)
    return np.clip(indices, 0, n - 1)


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute effective sample size."""
    return 1.0 / np.sum(weights**2)


class ParticleFilter:
    """
    Bootstrap Particle Filter.

    Example:
        def transition(x):
            return x + np.random.normal(0, 0.1, len(x))

        def likelihood(y, x):
            return np.exp(-0.5 * (y - x)**2 / 0.5**2)

        pf = ParticleFilter(
            n_particles=1000,
            transition_fn=transition,
            likelihood_fn=likelihood,
        )
        estimates = pf.filter(observations)
    """

    def __init__(
        self,
        n_particles: int,
        transition_fn: Callable[[np.ndarray], np.ndarray],
        likelihood_fn: Callable[[float, np.ndarray], np.ndarray],
        resample_threshold: float = 0.5,
    ):
        self.n_particles = n_particles
        self.transition = transition_fn
        self.likelihood = likelihood_fn
        self.resample_threshold = resample_threshold

    def initialize(self, prior_sample: Callable[[int], np.ndarray]) -> ParticleState:
        """Initialize particles from prior."""
        particles = prior_sample(self.n_particles)
        weights = np.ones(self.n_particles) / self.n_particles

        return ParticleState(
            particles=particles,
            weights=weights,
            estimate=np.mean(particles),
            variance=np.var(particles),
            ess=self.n_particles,
        )

    def step(self, state: ParticleState, observation: float) -> ParticleState:
        """One step of particle filter."""
        particles = self.transition(state.particles)
        likelihoods = self.likelihood(observation, particles)

        weights = state.weights * likelihoods
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            weights = np.ones(self.n_particles) / self.n_particles

        ess = effective_sample_size(weights)

        if ess < self.resample_threshold * self.n_particles:
            indices = systematic_resample(weights)
            particles = particles[indices]
            weights = np.ones(self.n_particles) / self.n_particles
            ess = self.n_particles

        estimate = np.sum(weights * particles)
        variance = np.sum(weights * (particles - estimate) ** 2)

        return ParticleState(
            particles=particles,
            weights=weights,
            estimate=estimate,
            variance=variance,
            ess=ess,
        )

    def filter(
        self,
        observations: np.ndarray,
        prior_sample: Callable[[int], np.ndarray],
    ) -> List[ParticleState]:
        """Run particle filter on observations."""
        state = self.initialize(prior_sample)
        states = []

        for y in observations:
            if not np.isnan(y):
                state = self.step(state, y)
            states.append(state)

        return states
