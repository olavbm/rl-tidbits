"""Shared test fixtures.

Key optimization: JAX recompiles for every unique config shape. By sharing
a single tiny config across all training-related tests, we compile once.
"""

import jax
import pytest

from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.types import EnvConfig
from jax_boids.ppo import create_train_state


# Minimal config for training/rollout tests — smallest possible to minimize compile time
TINY_CONFIG = EnvConfig(
    n_predators=1,
    n_prey=1,
    world_size=10.0,
    max_steps=20,
    k_nearest_same=0,
    k_nearest_enemy=1,
    separation_weight=0.0,
    alignment_weight=0.0,
    cohesion_weight=0.0,
    capture_radius=2.0,
)


@pytest.fixture
def tiny_env():
    """Tiny environment for fast compilation."""
    return PredatorPreyEnv(TINY_CONFIG)


@pytest.fixture
def tiny_train_state(tiny_env):
    """Train state matching the tiny env."""
    key = jax.random.PRNGKey(42)
    return create_train_state(key, tiny_env.observation_size, tiny_env.action_size, 3e-4, 0.5)
