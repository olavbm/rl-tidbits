"""Shared PPO utilities for training.

Contains core PPO components used by both multi-agent and single-agent training.
"""

from typing import Dict, NamedTuple

import chex
import distrax
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from jax_boids.networks import ActorCritic


from typing import Callable

from jax_boids.networks import NetworkOutput

PolicyFunction = Callable[[chex.Array, TrainState], NetworkOutput]


class Transition(NamedTuple):
    """Single transition for PPO."""

    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    log_prob: chex.Array
    value: chex.Array


class BaseTrainConfig(NamedTuple):
    """Base training hyperparameters shared by all training scripts."""

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_steps: int = 128  # steps per rollout
    n_epochs: int = 4  # PPO epochs per update
    n_minibatches: int = 4
    total_timesteps: int = 1_000_000
    n_envs: int = 32  # parallel environments


def make_distribution(action_mean: chex.Array, action_logstd: chex.Array) -> distrax.Distribution:
    """Create distribution from network outputs, handling batched inputs."""
    action_std = jnp.exp(action_logstd)
    action_std = jnp.broadcast_to(action_std, action_mean.shape)
    return distrax.MultivariateNormalDiag(action_mean, action_std)


def create_train_state(
    key: chex.PRNGKey,
    obs_size: int,
    action_size: int,
    lr: float,
) -> TrainState:
    """Initialize network and optimizer."""
    network = ActorCritic(action_dim=action_size)
    dummy_obs = jnp.zeros((obs_size,))
    params = network.init(key, dummy_obs)
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(lr),
    )
    return TrainState.create(apply_fn=network.apply, params=params, tx=tx)


def compute_gae(
    rewards: chex.Array,
    values: chex.Array,
    dones: chex.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[chex.Array, chex.Array]:
    """Compute GAE advantages and returns.

    Args:
        rewards: [T, N] rewards
        values: [T+1, N] values (including bootstrap)
        dones: [T, N] done flags
        gamma: discount factor
        gae_lambda: GAE lambda

    Returns:
        advantages: [T, N]
        returns: [T, N]
    """

    def scan_fn(carry, inputs):
        gae = carry
        reward, value, next_value, done = inputs
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        return gae, gae

    # Reverse scan for GAE
    _, advantages = jax.lax.scan(
        scan_fn,
        jnp.zeros_like(rewards[0]),
        (rewards[::-1], values[:-1][::-1], values[1:][::-1], dones[::-1]),
    )
    advantages = advantages[::-1]
    returns = advantages + values[:-1]

    return advantages, returns


def ppo_loss(
    params,
    apply_fn,
    obs: chex.Array,
    actions: chex.Array,
    old_log_probs: chex.Array,
    advantages: chex.Array,
    returns: chex.Array,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[chex.Array, Dict]:
    """Compute PPO loss."""
    out = jax.vmap(lambda o: apply_fn(params, o))(obs)
    pi = make_distribution(out.action_mean, out.action_logstd)
    values = out.value
    log_probs = pi.log_prob(actions)

    # Policy loss with clipping
    ratio = jnp.exp(log_probs - old_log_probs)
    clipped_ratio = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = -jnp.minimum(ratio * advantages, clipped_ratio * advantages).mean()

    # Value loss
    value_loss = ((values - returns) ** 2).mean()

    # Entropy bonus
    entropy = pi.entropy().mean()

    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": ((ratio - 1) - jnp.log(ratio)).mean(),
    }

    return total_loss, metrics


def ppo_update(
    train_state: TrainState,
    transitions: Transition,
    key: chex.PRNGKey,
    gamma: float,
    gae_lambda: float,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    n_epochs: int,
    n_minibatches: int,
) -> tuple[TrainState, Dict]:
    """PPO update for a batch of transitions.

    Args:
        train_state: Current training state
        transitions: Batch of transitions with shapes [T, n_envs, n_agents, ...]
        key: Random key for shuffling
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_eps: PPO clipping epsilon
        vf_coef: Value function coefficient
        ent_coef: Entropy coefficient
        n_epochs: Number of PPO epochs
        n_minibatches: Number of minibatches

    Returns:
        Updated train_state and metrics dict
    """
    # transitions shapes: [T, n_envs, n_agents, ...]
    T = transitions.obs.shape[0]
    obs_flat = transitions.obs.reshape(T, -1, transitions.obs.shape[-1])
    actions_flat = transitions.action.reshape(T, -1, transitions.action.shape[-1])
    log_probs_flat = transitions.log_prob.reshape(T, -1)
    values_flat = transitions.value.reshape(T, -1)
    rewards_flat = transitions.reward.reshape(T, -1)
    dones_flat = transitions.done.reshape(T, -1)

    # Bootstrap values
    bootstrap_out = jax.vmap(lambda o: train_state.apply_fn(train_state.params, o))(obs_flat[-1])
    bootstrap_values = bootstrap_out.value
    values_with_bootstrap = jnp.concatenate([values_flat, bootstrap_values[None]], axis=0)

    advantages, returns = compute_gae(
        rewards_flat,
        values_with_bootstrap,
        dones_flat,
        gamma,
        gae_lambda,
    )

    # Flatten time dimension
    obs = obs_flat.reshape(-1, obs_flat.shape[-1])
    actions = actions_flat.reshape(-1, actions_flat.shape[-1])
    old_log_probs = log_probs_flat.flatten()
    advantages = advantages.flatten()
    returns = returns.flatten()

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch_size = obs.shape[0]
    minibatch_size = batch_size // n_minibatches

    def _epoch_step(train_state, key):
        perm = jax.random.permutation(key, batch_size)
        obs_shuffled = obs[perm]
        actions_shuffled = actions[perm]
        old_log_probs_shuffled = old_log_probs[perm]
        advantages_shuffled = advantages[perm]
        returns_shuffled = returns[perm]

        def _minibatch_step(train_state, idx):
            start = idx * minibatch_size
            mb_obs = jax.lax.dynamic_slice(
                obs_shuffled, (start, 0), (minibatch_size, obs.shape[-1])
            )
            mb_actions = jax.lax.dynamic_slice(
                actions_shuffled, (start, 0), (minibatch_size, actions.shape[-1])
            )
            mb_old_log_probs = jax.lax.dynamic_slice(
                old_log_probs_shuffled, (start,), (minibatch_size,)
            )
            mb_advantages = jax.lax.dynamic_slice(advantages_shuffled, (start,), (minibatch_size,))
            mb_returns = jax.lax.dynamic_slice(returns_shuffled, (start,), (minibatch_size,))

            def loss_fn(params):
                return ppo_loss(
                    params,
                    train_state.apply_fn,
                    mb_obs,
                    mb_actions,
                    mb_old_log_probs,
                    mb_advantages,
                    mb_returns,
                    clip_eps,
                    vf_coef,
                    ent_coef,
                )

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (_, metrics), grads = grad_fn(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, metrics

        train_state, metrics = jax.lax.scan(_minibatch_step, train_state, jnp.arange(n_minibatches))
        return train_state, metrics

    keys = jax.random.split(key, n_epochs)
    train_state, metrics = jax.lax.scan(_epoch_step, train_state, keys)
    metrics = jax.tree.map(lambda x: x.mean(), metrics)
    return train_state, metrics


def select_on_reset(reset_mask: chex.Array, new: chex.Array, old: chex.Array) -> chex.Array:
    """Select between new and old values based on reset mask.

    Broadcasts reset_mask to match array dimensions.

    Args:
        reset_mask: [n_envs] boolean mask
        new: New values (after reset)
        old: Old values (before reset)

    Returns:
        Selected values with same shape as old
    """
    if old.ndim == 1:
        mask = reset_mask
    elif old.ndim == 2:
        mask = reset_mask[:, None]
    else:
        mask = reset_mask[:, None, None]
    return jnp.where(mask, new, old)
