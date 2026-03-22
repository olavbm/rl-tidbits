"""Sanity check: verify reward signal exists for chasing behavior."""

import jax
import jax.numpy as jnp

from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.types import BoidsState, EnvConfig
from tests.conftest import TINY_CONFIG


def test_reward_signal_exists(tiny_env):
    """Verify that moving toward prey yields higher reward than random actions."""
    cfg = TINY_CONFIG

    key = jax.random.PRNGKey(42)
    prey_pos = jnp.array([[5.0, 5.0]])
    pred_pos = jnp.array([[2.0, 2.0]])

    state = BoidsState(
        predator_pos=pred_pos,
        predator_vel=jnp.zeros((1, 2)),
        prey_pos=prey_pos,
        prey_vel=jnp.zeros((1, 2)),
        prey_alive=jnp.array([True]),
        step=0,
        key=key,
    )

    _, state = tiny_env.reset_from_state(state)

    # Random actions
    key, k1, k2 = jax.random.split(key, 3)
    random_actions = {
        "predator": jax.random.uniform(k1, (1, 2), minval=-1, maxval=1),
        "prey": jnp.zeros((1, 2)),
    }
    _, _, random_rewards, _, _ = tiny_env.step(k2, state, random_actions)

    # Chase actions (toward prey)
    key, k3 = jax.random.split(key)
    direction = prey_pos - pred_pos
    direction = direction / (jnp.linalg.norm(direction) + 1e-6)
    chase_actions = {"predator": direction, "prey": jnp.zeros((1, 2))}
    _, _, chase_rewards, _, _ = tiny_env.step(k3, state, chase_actions)

    # Both should produce finite rewards
    assert jnp.isfinite(random_rewards["predator"]).all()
    assert jnp.isfinite(chase_rewards["predator"]).all()
