"""Verify PPO update actually changes the policy."""

import jax
import jax.numpy as jnp

from jax_boids.collector import PolicyConfig, PolicyType, RolloutConfig, collect_rollouts
from jax_boids.ppo import ppo_update
from tests.conftest import TINY_CONFIG


def test_ppo_update_changes_policy(tiny_env, tiny_train_state):
    """Verify that ppo_update actually changes the policy parameters."""
    policies = {
        "predator": PolicyConfig(PolicyType.LEARNED, train_state=tiny_train_state),
        "prey": PolicyConfig(PolicyType.RANDOM, noise_scale=0.2),
    }
    rollout_config = RolloutConfig(n_steps=8, n_envs=2)

    key = jax.random.PRNGKey(42)
    key, (transitions, _), obs, env_state = collect_rollouts(
        tiny_env, policies, TINY_CONFIG, rollout_config, key
    )

    test_obs = transitions["predator"].obs[0, 0, 0]
    initial_out = tiny_train_state.apply_fn(tiny_train_state.params, test_obs)
    initial_mean = initial_out.action_mean

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

    new_out = new_train_state.apply_fn(new_train_state.params, test_obs)
    change = jnp.abs(new_out.action_mean - initial_mean).mean()
    assert change > 1e-6, f"Policy didn't change! change={change}"
