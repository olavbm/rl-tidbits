"""Check advantage computation and value estimates."""

import jax
import jax.numpy as jnp

from jax_boids.collector import PolicyConfig, PolicyType, RolloutConfig, collect_rollouts
from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.ppo import compute_gae
from tests.conftest import TINY_CONFIG


def test_advantage_magnitude(tiny_env, tiny_train_state):
    """Check if advantages are reasonable."""
    policies = {
        "predator": PolicyConfig(PolicyType.LEARNED, train_state=tiny_train_state),
        "prey": PolicyConfig(PolicyType.RANDOM, noise_scale=0.2),
    }
    rollout_config = RolloutConfig(n_steps=8, n_envs=2)

    key = jax.random.PRNGKey(42)
    key, (transitions, _), obs, env_state = collect_rollouts(
        tiny_env, policies, TINY_CONFIG, rollout_config, key
    )

    rewards = transitions["predator"].reward
    values = transitions["predator"].value

    T = rewards.shape[0]
    rewards_flat = rewards.reshape(T, -1)
    values_flat = values.reshape(T, -1)
    dones_flat = transitions["predator"].done.reshape(T, -1)

    bootstrap_out = jax.vmap(lambda o: tiny_train_state.apply_fn(tiny_train_state.params, o))(
        transitions["predator"].obs[-1].reshape(-1, transitions["predator"].obs.shape[-1])
    )
    bootstrap_values = bootstrap_out.value
    values_with_bootstrap = jnp.concatenate([values_flat, bootstrap_values[None]], axis=0)

    advantages, returns = compute_gae(
        rewards_flat, values_with_bootstrap, dones_flat, gamma=0.99, gae_lambda=0.95
    )

    # Advantages should have meaningful variance (not all the same)
    assert advantages.std() > 0.01, f"Advantage std too small: {advantages.std()}"
