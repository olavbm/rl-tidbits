"""Check if value head updates during PPO."""

import jax
import jax.numpy as jnp

from jax_boids.collector import PolicyConfig, PolicyType, RolloutConfig, collect_rollouts
from jax_boids.ppo import ppo_update
from tests.conftest import TINY_CONFIG


def test_value_head_updates(tiny_env, tiny_train_state):
    """Verify value head actually changes after PPO update."""
    policies = {
        "predator": PolicyConfig(PolicyType.LEARNED, train_state=tiny_train_state),
        "prey": PolicyConfig(PolicyType.RANDOM, noise_scale=0.2),
    }
    rollout_config = RolloutConfig(n_steps=8, n_envs=2)

    key = jax.random.PRNGKey(42)
    key, (transitions, _), obs, env_state = collect_rollouts(
        tiny_env, policies, TINY_CONFIG, rollout_config, key
    )

    test_obs = transitions["predator"].obs[0].reshape(-1, transitions["predator"].obs.shape[-1])
    initial_values = jax.vmap(lambda o: tiny_train_state.apply_fn(tiny_train_state.params, o))(test_obs)
    initial_value_mean = initial_values.value.mean()

    key, k1 = jax.random.split(key)
    new_train_state, metrics, _, _ = ppo_update(
        tiny_train_state,
        transitions["predator"],
        k1,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        n_epochs=2,
        n_minibatches=2,
    )

    new_values = jax.vmap(lambda o: new_train_state.apply_fn(new_train_state.params, o))(test_obs)
    new_value_mean = new_values.value.mean()

    value_change = abs(float(new_value_mean - initial_value_mean))
    assert value_change > 0.001, f"Value head barely changed: {value_change}"
