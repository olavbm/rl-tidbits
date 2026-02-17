"""JAX-based boids environment for multi-agent self-play."""

from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.types import BoidsState

__all__ = ["PredatorPreyEnv", "BoidsState"]
