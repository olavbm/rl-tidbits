"""Check if value head updates during PPO."""

import jax
import jax.numpy as jnp

from jax_boids.collector import PolicyConfig, PolicyType, RolloutConfig, collect_rollouts
from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.types import EnvConfig
from jax_boids.ppo import create_train_state, ppo_update


def test_value_head_updates():
    """Verify value head actually changes after PPO update."""
    env_config = EnvConfig(n_predators=2, n_prey=2, max_steps=50)
    env = PredatorPreyEnv(env_config)

    # Create policy
    key = jax.random.PRNGKey(42)
    train_state = create_train_state(key, env.observation_size, env.action_size, 3e-4, 0.5)

    # Collect rollout
    policies = {
        "predator": PolicyConfig(PolicyType.LEARNED, train_state=train_state),
        "prey": PolicyConfig(PolicyType.RANDOM, noise_scale=0.2),
    }
    rollout_config = RolloutConfig(n_steps=16, n_envs=4)

    key, (transitions, _), obs, env_state = collect_rollouts(
        env, policies, env_config, rollout_config, key
    )

    # Get initial value predictions
    test_obs = transitions["predator"].obs[0].reshape(-1, transitions["predator"].obs.shape[-1])
    initial_values = jax.vmap(lambda o: train_state.apply_fn(train_state.params, o))(test_obs)
    initial_value_mean = initial_values.value.mean()

    print(f"Initial value predictions (mean): {initial_value_mean:.4f}")
    print(f"Actual rewards (mean): {transitions['predator'].reward.mean():.4f}")

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

    # Get new value predictions for same observations
    new_values = jax.vmap(lambda o: new_train_state.apply_fn(new_train_state.params, o))(test_obs)
    new_value_mean = new_values.value.mean()

    print(f"\nNew value predictions (mean): {new_value_mean:.4f}")
    print(f"Value change: {new_value_mean - initial_value_mean:.4f}")
    print(f"Value loss: {metrics['value_loss']:.4f}")

    # Debug: check returns computation
    from jax_boids.ppo import compute_gae

    T = transitions["predator"].obs.shape[0]
    rewards_flat = transitions["predator"].reward.reshape(T, -1)
    values_flat = transitions["predator"].value.reshape(T, -1)
    dones_flat = transitions["predator"].done.reshape(T, -1)

    bootstrap_out = jax.vmap(lambda o: train_state.apply_fn(train_state.params, o))(
        transitions["predator"].obs[-1].reshape(-1, transitions["predator"].obs.shape[-1])
    )
    bootstrap_values = bootstrap_out.value
    values_with_bootstrap = jnp.concatenate([values_flat, bootstrap_values[None]], axis=0)

    advantages, returns = compute_gae(
        rewards_flat, values_with_bootstrap, dones_flat, gamma=0.99, gae_lambda=0.95
    )

    print(f"\nReturns stats: mean={returns.mean():.4f}, std={returns.std():.4f}")
    print(f"Returns range: [{returns.min():.2f}, {returns.max():.2f}]")
    print(f"Values stats: mean={values_flat.mean():.4f}, std={values_flat.std():.4f}")
    print(f"Raw MSE (values - returns)^2: {((values_flat - returns) ** 2).mean():.4f}")

    # Check per-timestep returns
    print("\nPer-timestep returns (first env):")
    for t in range(min(5, returns.shape[0])):
        print(f"  t={t}: return={returns[t, 0]:.2f}, reward={rewards_flat[t, 0]:.2f}")

    # The value head SHOULD move toward the actual returns
    value_change = new_value_mean - initial_value_mean
    if abs(value_change) < 0.01:
        print("⚠ WARNING: Value head barely changed!")
    else:
        print(f"✓ Value head changed: {value_change:.4f}")


if __name__ == "__main__":
    test_value_head_updates()
