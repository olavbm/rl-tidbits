"""Check gradient norms during PPO update."""

import jax
import jax.numpy as jnp
import optax

from jax_boids.collector import PolicyConfig, PolicyType, RolloutConfig, collect_rollouts
from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.types import EnvConfig
from jax_boids.ppo import create_train_state, ppo_loss, make_distribution, compute_gae


def test_gradient_norms():
    """Check if gradients are reasonable or being clipped."""
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

    # Prepare data for PPO loss
    pred_trans = transitions["predator"]
    T = pred_trans.obs.shape[0]
    obs_flat = pred_trans.obs.reshape(T, -1, pred_trans.obs.shape[-1])
    actions_flat = pred_trans.action.reshape(T, -1, pred_trans.action.shape[-1])
    log_probs_flat = pred_trans.log_prob.reshape(T, -1)
    values_flat = pred_trans.value.reshape(T, -1)
    rewards_flat = pred_trans.reward.reshape(T, -1)
    dones_flat = pred_trans.done.reshape(T, -1)

    # Bootstrap
    bootstrap_out = jax.vmap(lambda o: train_state.apply_fn(train_state.params, o))(obs_flat[-1])
    bootstrap_values = bootstrap_out.value
    values_with_bootstrap = jnp.concatenate([values_flat, bootstrap_values[None]], axis=0)

    advantages, returns = compute_gae(
        rewards_flat, values_with_bootstrap, dones_flat, gamma=0.99, gae_lambda=0.95
    )

    # Flatten
    obs_flat = obs_flat.reshape(-1, obs_flat.shape[-1])
    actions_flat = actions_flat.reshape(-1, actions_flat.shape[-1])
    old_log_probs = log_probs_flat.flatten()
    advantages = advantages.flatten()
    returns = returns.flatten()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Compute loss and gradients
    total_loss, metrics = ppo_loss(
        train_state.params,
        train_state.apply_fn,
        obs_flat,
        actions_flat,
        old_log_probs,
        advantages,
        returns,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
    )

    def loss_only(params):
        return ppo_loss(
            params,
            train_state.apply_fn,
            obs_flat,
            actions_flat,
            old_log_probs,
            advantages,
            returns,
            clip_eps=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
        )[0]

    grads = jax.grad(loss_only)(train_state.params)

    # Compute gradient norm
    grad_norm = optax.global_norm(grads)
    print(f"Total loss: {total_loss:.4f}")
    print(f"Policy loss: {metrics['policy_loss']:.4f}")
    print(f"Value loss: {metrics['value_loss']:.4f}")
    print(f"Entropy: {metrics['entropy']:.4f}")
    print(f"Gradient norm: {grad_norm:.4f}")

    # Check individual param grad norms
    for name, grad in grads.items():
        param_norm = optax.global_norm(grad)
        print(f"  {name}: grad_norm={param_norm:.4f}")

    # The optimizer clips at 0.5
    if grad_norm > 0.5:
        print(f"\n⚠ Gradient norm {grad_norm:.4f} > 0.5, will be clipped!")
        print(f"  This means updates are smaller than they should be.")
    else:
        print(f"\n✓ Gradient norm is within clipping threshold")


if __name__ == "__main__":
    test_gradient_norms()
