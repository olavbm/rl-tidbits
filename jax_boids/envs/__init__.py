"""Boids environments."""

from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.rewards import compute_predator_rewards, compute_prey_rewards
from jax_boids.envs.types import BoidsState

__all__ = [
    "PredatorPreyEnv",
    "BoidsState",
    "compute_predator_rewards",
    "compute_prey_rewards",
]
