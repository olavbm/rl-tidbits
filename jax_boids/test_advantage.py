"""Check advantage computation and value estimates."""

import jax
import jax.numpy as jnp

from jax_boids.collector import PolicyConfig, PolicyType, RolloutConfig, collect_rollouts
from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.types import EnvConfig
from jax_boids.ppo import compute_gae, create_train_state


def test_advantage_magnitude():
    """Check if advantages are reasonable."""
    env_config = EnvConfig(n_predators=2, n_prey=2, max_steps=50)
    env = PredatorPreyEnv(env_config)

    # Create policy
    key = jax.random.PRNGKey(42)
    train_state = create_train_state(key, env.observation_size, env.action_size, 3e-4)

    # Collect rollout
    policies = {
        "predator": PolicyConfig(PolicyType.LEARNED, train_state=train_state),
        "prey": PolicyConfig(PolicyType.RANDOM, noise_scale=0.2),
    }
    rollout_config = RolloutConfig(n_steps=16, n_envs=4)

    key, (transitions, _), obs, env_state = collect_rollouts(
        env, policies, env_config, rollout_config, key
    )

    # Get rewards and values
    rewards = transitions["predator"].reward  # [T, n_envs, n_agents]
    values = transitions["predator"].value  # [T, n_envs, n_agents]

    print(
        f"Reward stats: min={rewards.min():.4f}, max={rewards.max():.4f}, mean={rewards.mean():.4f}"
    )
    print(f"Value stats: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")

    # Flatten for GAE computation
    T = rewards.shape[0]
    rewards_flat = rewards.reshape(T, -1)  # [T, n_envs*n_agents]
    values_flat = values.reshape(T, -1)  # [T, n_envs*n_agents]
    dones_flat = transitions["predator"].done.reshape(T, -1)

    # Bootstrap
    bootstrap_out = jax.vmap(lambda o: train_state.apply_fn(train_state.params, o))(
        transitions["predator"].obs[-1].reshape(-1, transitions["predator"].obs.shape[-1])
    )
    bootstrap_values = bootstrap_out.value
    values_with_bootstrap = jnp.concatenate([values_flat, bootstrap_values[None]], axis=0)

    # Compute GAE
    advantages, returns = compute_gae(
        rewards_flat, values_with_bootstrap, dones_flat, gamma=0.99, gae_lambda=0.95
    )

    print("\nAdvantage stats:")
    print(f"  min={advantages.min():.4f}, max={advantages.max():.4f}")
    print(f"  mean={advantages.mean():.4f}, std={advantages.std():.4f}")
    print(
        f"Return stats: min={returns.min():.4f}, max={returns.max():.4f}, mean={returns.mean():.4f}"
    )

    # Check if values are close to actual returns
    # (they shouldn't be at initialization, but let's see how wrong they are)
    value_error = (values_flat - returns).mean()
    print(f"\nMean value error: {value_error:.4f}")

    # The key question: are advantages meaningful?
    # If std is very small, all advantages are similar → no learning signal
    if advantages.std() < 0.1:
        print("⚠ WARNING: Advantage std is very small - no learning signal!")
    else:
        print(f"✓ Advantage std is reasonable: {advantages.std():.4f}")


if __name__ == "__main__":
    test_advantage_magnitude()
