"""Check gradient norms during PPO update."""

import jax
import jax.numpy as jnp
import optax

from jax_boids.collector import PolicyConfig, PolicyType, RolloutConfig, collect_rollouts
from jax_boids.ppo import compute_gae, ppo_loss
from tests.conftest import TINY_CONFIG


def test_gradient_norms(tiny_env, tiny_train_state):
    """Check if gradients are reasonable."""
    policies = {
        "predator": PolicyConfig(PolicyType.LEARNED, train_state=tiny_train_state),
        "prey": PolicyConfig(PolicyType.RANDOM, noise_scale=0.2),
    }
    rollout_config = RolloutConfig(n_steps=8, n_envs=2)

    key = jax.random.PRNGKey(42)
    key, (transitions, _), obs, env_state = collect_rollouts(
        tiny_env, policies, TINY_CONFIG, rollout_config, key
    )

    pred_trans = transitions["predator"]
    T = pred_trans.obs.shape[0]
    obs_flat = pred_trans.obs.reshape(T, -1, pred_trans.obs.shape[-1])
    actions_flat = pred_trans.action.reshape(T, -1, pred_trans.action.shape[-1])
    log_probs_flat = pred_trans.log_prob.reshape(T, -1)
    values_flat = pred_trans.value.reshape(T, -1)
    rewards_flat = pred_trans.reward.reshape(T, -1)
    dones_flat = pred_trans.done.reshape(T, -1)

    bootstrap_out = jax.vmap(lambda o: tiny_train_state.apply_fn(tiny_train_state.params, o))(obs_flat[-1])
    bootstrap_values = bootstrap_out.value
    values_with_bootstrap = jnp.concatenate([values_flat, bootstrap_values[None]], axis=0)

    advantages, returns = compute_gae(
        rewards_flat, values_with_bootstrap, dones_flat, gamma=0.99, gae_lambda=0.95
    )

    obs_flat = obs_flat.reshape(-1, obs_flat.shape[-1])
    actions_flat = actions_flat.reshape(-1, actions_flat.shape[-1])
    old_log_probs = log_probs_flat.flatten()
    advantages = advantages.flatten()
    returns = returns.flatten()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def loss_only(params):
        return ppo_loss(
            params, tiny_train_state.apply_fn, obs_flat, actions_flat,
            old_log_probs, advantages, returns,
            clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
        )[0]

    grads = jax.grad(loss_only)(tiny_train_state.params)
    grad_norm = optax.global_norm(grads)

    assert jnp.isfinite(grad_norm), f"Non-finite gradient norm: {grad_norm}"
    assert grad_norm > 0, "Zero gradients"
