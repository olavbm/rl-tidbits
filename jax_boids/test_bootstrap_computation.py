"""Test to verify bootstrap values are computed and used correctly in GAE."""

import jax
import jax.numpy as jnp

from jax_boids.collector import PolicyConfig, PolicyType, RolloutConfig, collect_rollouts
from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.types import EnvConfig
from jax_boids.ppo import compute_gae, create_train_state


def test_bootstrap_computation():
    """Verify bootstrap values are computed and used in GAE."""

    env_config = EnvConfig(n_predators=2, n_prey=2, max_steps=50)
    env = PredatorPreyEnv(env_config)
    rollout_config = RolloutConfig(n_steps=4, n_envs=2)

    # Initialize policy
    key = jax.random.PRNGKey(42)
    k1, key = jax.random.split(key, 2)
    train_state = create_train_state(k1, env.observation_size, env.action_size, 3e-4)

    policies = {
        "predator": PolicyConfig(PolicyType.LEARNED, train_state=train_state, noise_scale=0.0),
        "prey": PolicyConfig(PolicyType.RANDOM, noise_scale=0.3),
    }

    # Collect rollouts
    key, (transitions, infos), final_obs, env_state = collect_rollouts(
        env, policies, env_config, rollout_config, key
    )

    # Get predator transitions
    trans = transitions["predator"]
    T = trans.obs.shape[0]  # n_steps
    n_envs = trans.obs.shape[1]
    n_agents = trans.obs.shape[2]

    print(f"Transitions shape: [T={T}, n_envs={n_envs}, n_agents={n_agents}]")
    print(f"Rewards: mean={trans.reward.mean():.4f}, std={trans.reward.std():.4f}")
    print(f"Values: mean={trans.value.mean():.4f}, std={trans.value.std():.4f}")
    print(f"Dones: sum={trans.done.sum()}")

    # Flatten for GAE computation
    obs_flat = trans.obs.reshape(T, -1, trans.obs.shape[-1])
    values_flat = trans.value.reshape(T, -1)
    rewards_flat = trans.reward.reshape(T, -1)
    dones_flat = trans.done.reshape(T, -1)

    # Compute bootstrap values from final observations
    bootstrap_out = jax.vmap(lambda o: train_state.apply_fn(train_state.params, o))(obs_flat[-1])
    bootstrap_values = bootstrap_out.value

    print(
        f"\nBootstrap values: mean={bootstrap_values.mean():.4f}, std={bootstrap_values.std():.4f}"
    )
    print(
        f"Last timestep values: mean={values_flat[-1].mean():.4f}, std={values_flat[-1].std():.4f}"
    )

    # Concatenate with bootstrap
    values_with_bootstrap = jnp.concatenate([values_flat, bootstrap_values[None]], axis=0)
    print(f"Values with bootstrap shape: {values_with_bootstrap.shape}")

    # Compute GAE
    advantages, returns = compute_gae(
        rewards_flat,
        values_with_bootstrap,
        dones_flat,
        gamma=0.99,
        gae_lambda=0.95,
    )

    print(f"\nAdvantages: mean={advantages.mean():.4f}, std={advantages.std():.4f}")
    print(f"Returns: mean={returns.mean():.4f}, std={returns.std():.4f}")

    # Verify bootstrap is being used by checking last timestep returns
    # For the last timestep, returns should incorporate bootstrap value
    last_returns = returns[-1]  # [n_envs * n_agents]
    last_values = values_flat[-1]
    last_rewards = rewards_flat[-1]
    last_dones = dones_flat[-1]

    # Expected returns for last timestep (simplified, ignoring GAE lambda)
    # V_target = r + gamma * bootstrap * (1 - done)
    expected_last_returns = last_rewards + 0.99 * bootstrap_values * (1 - last_dones)

    print("\nLast timestep analysis:")
    print(f"  Rewards: mean={last_rewards.mean():.4f}")
    print(f"  Values: mean={last_values.mean():.4f}")
    print(f"  Bootstrap: mean={bootstrap_values.mean():.4f}")
    print(f"  Returns: mean={last_returns.mean():.4f}")
    print(f"  Expected (r + gamma*V_bootstrap): mean={expected_last_returns.mean():.4f}")

    # Check if bootstrap is actually influencing returns
    # If bootstrap is being used, returns should be closer to expected_last_returns
    # than to just rewards + gamma * last_values
    expected_without_bootstrap = last_rewards + 0.99 * last_values * (1 - last_dones)

    diff_with_bootstrap = jnp.abs(last_returns - expected_last_returns).mean()
    diff_without_bootstrap = jnp.abs(last_returns - expected_without_bootstrap).mean()

    print(f"\nDifference from expected (with bootstrap): {diff_with_bootstrap:.4f}")
    print(f"Difference from expected (without bootstrap): {diff_without_bootstrap:.4f}")

    # The returns should be closer to the version with bootstrap
    # (allowing for GAE lambda effects)
    assert diff_with_bootstrap < diff_without_bootstrap, (
        f"Bootstrap doesn't seem to be used in returns computation. "
        f"Diff with bootstrap: {diff_with_bootstrap:.4f}, without: {diff_without_bootstrap:.4f}"
    )

    print("\n✓ Bootstrap values are being used in GAE computation")


if __name__ == "__main__":
    test_bootstrap_computation()
