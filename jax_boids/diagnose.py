"""Quick diagnostic to check if training is working."""

import jax
import jax.numpy as jnp

from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.types import EnvConfig
from jax_boids.ppo import Transition, create_train_state, make_distribution, ppo_update


def run_diagnostics():
    """Run a single update and print diagnostics."""
    print("=" * 60)
    print("DIAGNOSTIC: Checking training components")
    print("=" * 60)

    # Setup
    env_config = EnvConfig(n_predators=2, n_prey=3)
    env = PredatorPreyEnv(env_config)
    n_envs = 4
    n_steps = 16
    key = jax.random.PRNGKey(42)

    k1, k2, k3, key = jax.random.split(key, 4)

    # Create train states
    pred_state = create_train_state(k1, env.observation_size, env.action_size, 3e-4)
    prey_state = create_train_state(k2, env.observation_size, env.action_size, 3e-4)

    # Reset environments
    env_keys = jax.random.split(k3, n_envs)
    obs, env_states = jax.vmap(env.reset)(env_keys)

    print("\n1. Environment setup:")
    print(f"   Observation size: {env.observation_size}")
    print(f"   Action size: {env.action_size}")
    print(f"   n_predators: {env_config.n_predators}")
    print(f"   n_prey: {env_config.n_prey}")
    print(f"   n_envs: {n_envs}")

    # Collect a few steps manually
    print(f"\n2. Collecting {n_steps} steps...")
    transitions_pred = []
    transitions_prey = []

    for step in range(n_steps):
        key, k1, k2, k3 = jax.random.split(key, 4)

        # Flatten obs
        pred_obs_flat = obs["predator"].reshape(-1, env.observation_size)
        prey_obs_flat = obs["prey"].reshape(-1, env.observation_size)

        # Get network outputs
        pred_out = jax.vmap(lambda o: pred_state.apply_fn(pred_state.params, o))(pred_obs_flat)
        prey_out = jax.vmap(lambda o: prey_state.apply_fn(prey_state.params, o))(prey_obs_flat)

        # Sample actions
        pred_pi = make_distribution(pred_out.action_mean, pred_out.action_logstd)
        prey_pi = make_distribution(prey_out.action_mean, prey_out.action_logstd)

        pred_actions_flat = pred_pi.sample(seed=k1)
        prey_actions_flat = prey_pi.sample(seed=k2)
        pred_log_probs_flat = pred_pi.log_prob(pred_actions_flat)
        prey_log_probs_flat = prey_pi.log_prob(prey_actions_flat)

        # Reshape
        n_pred = env_config.n_predators
        n_prey = env_config.n_prey
        pred_actions = pred_actions_flat.reshape(n_envs, n_pred, -1)
        prey_actions = prey_actions_flat.reshape(n_envs, n_prey, -1)
        pred_log_probs = pred_log_probs_flat.reshape(n_envs, n_pred)
        prey_log_probs = prey_log_probs_flat.reshape(n_envs, n_prey)
        pred_values = pred_out.value.reshape(n_envs, n_pred)
        prey_values = prey_out.value.reshape(n_envs, n_prey)

        # Step environment
        actions = {"predator": pred_actions, "prey": prey_actions}
        step_keys = jax.random.split(k3, n_envs)
        next_obs, env_states_new, rewards, dones, info = jax.vmap(env.step)(
            step_keys, env_states, actions
        )

        # Store transitions
        transitions_pred.append(
            Transition(
                obs=obs["predator"],
                action=pred_actions,
                reward=rewards["predator"],
                done=dones["predator"],
                log_prob=pred_log_probs,
                value=pred_values,
            )
        )
        transitions_prey.append(
            Transition(
                obs=obs["prey"],
                action=prey_actions,
                reward=rewards["prey"],
                done=dones["prey"],
                log_prob=prey_log_probs,
                value=prey_values,
            )
        )

        # Handle resets
        reset_mask = dones["__all__"]

        def _select(new, old):
            # Broadcast reset_mask to match shape
            mask = reset_mask.reshape(-1, *([1] * (old.ndim - 1)))
            return jnp.where(mask, new, old)

        obs = jax.tree.map(_select, next_obs, obs)
        env_states = jax.tree.map(_select, env_states_new, env_states)

    # Stack transitions
    transitions_pred = jax.tree.map(lambda *args: jnp.stack(args), *transitions_pred)
    transitions_prey = jax.tree.map(lambda *args: jnp.stack(args), *transitions_prey)

    # Check rewards
    print("\n3. Reward statistics:")
    print(
        f"   Predator rewards: mean={transitions_pred.reward.mean():.4f}, "
        f"std={transitions_pred.reward.std():.4f}, "
        f"min={transitions_pred.reward.min():.4f}, "
        f"max={transitions_pred.reward.max():.4f}"
    )
    print(
        f"   Prey rewards: mean={transitions_prey.reward.mean():.4f}, "
        f"std={transitions_prey.reward.std():.4f}, "
        f"min={transitions_prey.reward.min():.4f}, "
        f"max={transitions_prey.reward.max():.4f}"
    )
    print(f"   Prey alive: mean={info['prey_alive'].mean():.1f}")

    # Check entropy (policy randomness)
    print("\n4. Policy entropy (higher = more exploration):")
    pred_std = jnp.exp(pred_out.action_logstd)
    prey_std = jnp.exp(prey_out.action_logstd)
    print(f"   Predator action std: {pred_std.mean():.4f}")
    print(f"   Prey action std: {prey_std.mean():.4f}")

    # Run PPO update
    print("\n5. Running PPO update...")
    k1, k2 = jax.random.split(key)

    pred_state_new, pred_metrics = ppo_update(
        pred_state,
        transitions_pred,
        k1,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        n_epochs=4,
        n_minibatches=4,
    )
    prey_state_new, prey_metrics = ppo_update(
        prey_state,
        transitions_prey,
        k2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        n_epochs=4,
        n_minibatches=4,
    )

    print("\n6. PPO metrics:")
    print(
        f"   Predator - policy_loss: {pred_metrics['policy_loss']:.6f}, "
        f"value_loss: {pred_metrics['value_loss']:.6f}, "
        f"entropy: {pred_metrics['entropy']:.6f}, "
        f"approx_kl: {pred_metrics['approx_kl']:.6f}"
    )
    print(
        f"   Prey - policy_loss: {prey_metrics['policy_loss']:.6f}, "
        f"value_loss: {prey_metrics['value_loss']:.6f}, "
        f"entropy: {prey_metrics['entropy']:.6f}, "
        f"approx_kl: {prey_metrics['approx_kl']:.6f}"
    )

    # Check GAE and value predictions
    print("\n6. GAE and value analysis:")

    # Compute GAE manually for inspection
    bootstrap_out = jax.vmap(lambda o: pred_state.apply_fn(pred_state.params, o))(
        transitions_pred.obs[-1].reshape(-1, env.observation_size)
    )
    bootstrap_values = bootstrap_out.value  # Shape: (n_envs * n_pred,)

    all_values = jnp.concatenate(
        [transitions_pred.value.reshape(n_steps, -1), bootstrap_values[None]], axis=0
    )

    # Compute returns manually
    rewards_flat = transitions_pred.reward.reshape(n_steps, -1)
    dones_flat = transitions_pred.done.reshape(n_steps, -1)

    # Simple discounted return (no GAE)
    returns_simple = jnp.zeros_like(rewards_flat)
    carry = jnp.zeros(rewards_flat.shape[1:])
    for t in range(n_steps - 1, -1, -1):
        carry = rewards_flat[t] + 0.99 * all_values[t + 1] * (1 - dones_flat[t])
        returns_simple = returns_simple.at[t].set(carry)

    print(
        f"   Value predictions: mean={all_values[:-1].mean():.4f}, std={all_values[:-1].std():.4f}"
    )
    print(
        f"   Bootstrap values: mean={bootstrap_values.mean():.4f}, std={bootstrap_values.std():.4f}"
    )
    print(f"   Actual rewards: mean={rewards_flat.mean():.4f}, std={rewards_flat.std():.4f}")
    print(f"   Target returns: mean={returns_simple.mean():.4f}, std={returns_simple.std():.4f}")
    print(f"   Value error (MSE): {((all_values[:-1] - returns_simple) ** 2).mean():.4f}")
    returns_mean = returns_simple.mean()
    value_mean = all_values[:-1].mean() + 1e-8
    print(f"   Scale mismatch: returns_mean / value_mean = {returns_mean / value_mean:.2f}x")

    # Check gradients
    print("\n7. Gradient check:")
    print("   (Gradients computed via ppo_update)")
    print(f"   Policy loss non-zero: {pred_metrics['policy_loss'] != 0}")
    print(f"   Value loss non-zero: {pred_metrics['value_loss'] != 0}")

    # Summary
    print(f"\n{'=' * 60}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'=' * 60}")

    issues = []

    if transitions_pred.reward.std() < 1e-6:
        issues.append("⚠️ Predator rewards are constant (no signal)")
    if transitions_prey.reward.std() < 1e-6:
        issues.append("⚠️ Prey rewards are constant (no signal)")

    if pred_metrics["policy_loss"] == 0:
        issues.append("⚠️ Policy loss is zero (not learning)")
    if pred_metrics["value_loss"] == 0:
        issues.append("⚠️ Value loss is zero (not learning)")

    if pred_std.mean() < 0.01:
        issues.append("⚠️ Policy is too deterministic (not exploring)")

    if pred_metrics["approx_kl"] > 1.0:
        issues.append("⚠️ KL divergence too high (policy changing too fast)")

    if not issues:
        print("✅ All checks passed - training should work!")
    else:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   {issue}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    run_diagnostics()
