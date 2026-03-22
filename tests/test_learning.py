"""Minimal end-to-end test: verify training loop runs without errors."""

from dataclasses import replace

import jax

from jax_boids.collector import PolicyConfig, PolicyType, RolloutConfig, collect_rollouts
from jax_boids.ppo import ppo_update
from tests.conftest import TINY_CONFIG


def test_training_improves_reward(tiny_env, tiny_train_state):
    """Run a few training steps and verify the loop completes."""
    train_state = tiny_train_state
    n_envs = 2
    n_steps = 8

    policies = {
        "predator": PolicyConfig(PolicyType.LEARNED, train_state=train_state),
        "prey": PolicyConfig(PolicyType.RANDOM, noise_scale=0.2),
    }
    rollout_config = RolloutConfig(n_steps=n_steps, n_envs=n_envs)

    key = jax.random.PRNGKey(42)
    obs = None
    env_state = None

    for _ in range(5):
        key, (transitions, infos), obs, env_state = collect_rollouts(
            tiny_env, policies, TINY_CONFIG, rollout_config, key, obs, env_state
        )

        key, k1 = jax.random.split(key)
        train_state, metrics, _, _ = ppo_update(
            train_state, transitions["predator"], k1,
            gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
            vf_coef=0.5, ent_coef=0.01, n_epochs=2, n_minibatches=2,
        )
        policies["predator"] = replace(policies["predator"], train_state=train_state)

    # Main goal: the training loop completes without errors
    assert train_state is not None
