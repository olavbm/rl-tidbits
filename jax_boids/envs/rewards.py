"""Reward computation for predator-prey environment.

Pure functions, directly unit-testable. Extracted from PredatorPreyEnv
so reward shaping can evolve independently.
"""

import chex
import jax.numpy as jnp


def compute_predator_rewards(
    n_captures: chex.Numeric, n_predators: int, dist_to_prey: chex.Array
) -> chex.Array:
    """Compute predator rewards from capture count and distance reduction.

    Adds dense reward shaping: reward for moving closer to nearest prey.

    Args:
        n_captures: scalar, number of prey captured this step
        n_predators: number of predators
        dist_to_prey: [n_predators] minimum distance to alive prey

    Returns:
        [n_predators] reward array
    """
    # Capture reward (shared equally)
    capture_reward = jnp.full(n_predators, n_captures * 10.0 / n_predators)

    # Dense shaping: reward for small distances (max reward when touching prey)
    # Increased from 20 to 50 to cover typical distances in 100x100 world
    distance_reward = jnp.where(
        dist_to_prey < 50.0,  # Reward distance threshold
        (50.0 - dist_to_prey) * 0.05,  # Max +2.5 per step
        0.0,
    )

    return capture_reward + distance_reward


def compute_prey_rewards(
    prey_pos: chex.Array,
    pred_pos: chex.Array,
    prey_alive: chex.Array,
    captures: chex.Array,
) -> chex.Array:
    """Compute prey rewards from survival, capture penalty, and distance.

    Args:
        prey_pos: [n_prey, 2] prey positions
        pred_pos: [n_pred, 2] predator positions
        prey_alive: [n_prey] bool mask (after captures applied)
        captures: [n_prey] bool mask of newly captured this step

    Returns:
        [n_prey] reward array
    """
    # Survival reward
    survival_reward = jnp.where(prey_alive, 0.1, 0.0)

    # Penalty for being caught
    caught_penalty = jnp.where(captures, -10.0, 0.0)

    # Reward for distance from predators
    prey_to_pred_dist = jnp.linalg.norm(
        prey_pos[:, None, :] - pred_pos[None, :, :], axis=-1
    )  # [n_prey, n_pred]
    min_dist_to_pred = jnp.min(prey_to_pred_dist, axis=1)  # [n_prey]
    distance_reward = jnp.where(prey_alive, 0.01 * min_dist_to_pred, 0.0)

    return survival_reward + caught_penalty + distance_reward
