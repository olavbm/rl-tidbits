"""Verify PPO update actually changes the policy."""

import jax
import jax.numpy as jnp

from jax_boids.collector import PolicyConfig, PolicyType, RolloutConfig, collect_rollouts
from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.types import EnvConfig
from jax_boids.ppo import create_train_state, ppo_update


def test_ppo_update_changes_policy():
    """Verify that ppo_update actually changes the policy parameters."""
    env_config = EnvConfig(n_predators=2, n_prey=2, max_steps=50)
    env = PredatorPreyEnv(env_config)

    # Create initial policy
    key = jax.random.PRNGKey(42)
    train_state = create_train_state(key, env.observation_size, env.action_size, 3e-4, 0.5)

    # Collect a small rollout
    policies = {
        "predator": PolicyConfig(PolicyType.LEARNED, train_state=train_state),
        "prey": PolicyConfig(PolicyType.RANDOM, noise_scale=0.2),
    }
    rollout_config = RolloutConfig(n_steps=16, n_envs=4)

    key, (transitions, _), obs, env_state = collect_rollouts(
        env, policies, env_config, rollout_config, key
    )

    # Sample an observation and get initial action
    test_obs = transitions["predator"].obs[0, 0, 0]  # First step, first env, first agent
    initial_out = train_state.apply_fn(train_state.params, test_obs)
    initial_mean = initial_out.action_mean

    print(f"Initial action mean: {initial_mean}")

    # Run PPO update
    key, k1 = jax.random.split(key)
    new_train_state, metrics, _, _ = ppo_update(
        train_state,
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

    # Get new action for same observation
    new_out = new_train_state.apply_fn(new_train_state.params, test_obs)
    new_mean = new_out.action_mean

    print(f"New action mean: {new_mean}")
    print(f"Policy loss: {metrics['policy_loss']:.6f}")
    print(f"Value loss: {metrics['value_loss']:.6f}")
    print(f"Entropy: {metrics['entropy']:.6f}")

    # Check if policy changed
    change = jnp.abs(new_mean - initial_mean).mean()
    print(f"Mean change in action: {change:.6f}")

    # The policy SHOULD change (even if randomly)
    assert change > 1e-6, f"Policy didn't change! change={change}"


if __name__ == "__main__":
    test_ppo_update_changes_policy()
