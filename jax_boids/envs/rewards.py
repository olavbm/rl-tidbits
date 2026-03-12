"""Reward computation for predator-prey environment.

Pure functions, directly unit-testable. Extracted from PredatorPreyEnv
so reward shaping can evolve independently.
"""

import chex
import jax.numpy as jnp

from jax_boids.envs.boids import wrapped_diff


def compute_predator_rewards(
    n_captures: chex.Numeric,
    n_predators: int,
    dist_to_prey: chex.Array,
    use_distance_reward: bool = True,
) -> chex.Array:
    """Compute predator rewards from capture count and distance reduction.

    Args:
        n_captures: scalar, number of prey captured this step
        n_predators: number of predators
        dist_to_prey: [n_predators] minimum distance to alive prey
        use_distance_reward: if False, only capture reward (no shaping)

    Returns:
        [n_predators] reward array
    """
    # Capture reward (shared equally, scaled by number of predators)
    capture_reward = jnp.full(n_predators, n_captures * 50.0 / n_predators)

    if not use_distance_reward:
        return capture_reward

    # Dense shaping: reward for small distances (max reward when touching prey)
    distance_reward = jnp.where(
        dist_to_prey < 50.0,
        (50.0 - dist_to_prey) * 0.05,
        0.0,
    )

    return capture_reward + distance_reward


def compute_prey_rewards(
    prey_pos: chex.Array,
    pred_pos: chex.Array,
    prey_alive: chex.Array,
    captures: chex.Array,
    world_size: float = 0.0,
) -> chex.Array:
    """Compute prey rewards from survival, capture penalty, and distance.

    Args:
        prey_pos: [n_prey, 2] prey positions
        pred_pos: [n_pred, 2] predator positions
        prey_alive: [n_prey] bool mask (after captures applied)
        captures: [n_prey] bool mask of newly captured this step
        world_size: if > 0, use toroidal wrapping

    Returns:
        [n_prey] reward array
    """
    # Survival reward
    survival_reward = jnp.where(prey_alive, 0.1, 0.0)

    # Penalty for being caught
    caught_penalty = jnp.where(captures, -10.0, 0.0)

    # Reward for distance from predators (wrapped)
    if world_size > 0:
        diff = wrapped_diff(prey_pos[:, None, :], pred_pos[None, :, :], world_size)
    else:
        diff = prey_pos[:, None, :] - pred_pos[None, :, :]
    prey_to_pred_dist = jnp.linalg.norm(diff, axis=-1)  # [n_prey, n_pred]
    min_dist_to_pred = jnp.min(prey_to_pred_dist, axis=1)  # [n_prey]
    distance_reward = jnp.where(prey_alive, 0.01 * min_dist_to_pred, 0.0)

    return survival_reward + caught_penalty + distance_reward
