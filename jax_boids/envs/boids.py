"""Core boids physics: separation, alignment, cohesion."""

import chex
import jax.numpy as jnp


def compute_distances(pos: chex.Array) -> chex.Array:
    """Compute pairwise distances between all agents.

    Args:
        pos: [n_agents, 2] positions

    Returns:
        [n_agents, n_agents] distance matrix
    """
    diff = pos[:, None, :] - pos[None, :, :]  # [n, n, 2]
    return jnp.linalg.norm(diff, axis=-1)  # [n, n]


def compute_separation(
    pos: chex.Array,
    dists: chex.Array,
    perception_radius: float,
) -> chex.Array:
    """Compute separation force: steer away from close neighbors.

    Args:
        pos: [n_agents, 2] positions
        dists: [n_agents, n_agents] precomputed distances
        perception_radius: distance within which to consider neighbors

    Returns:
        [n_agents, 2] separation forces
    """
    diff = pos[:, None, :] - pos[None, :, :]  # from other to self

    # Neighbors within perception radius (excluding self)
    neighbors = (dists < perception_radius) & (dists > 1e-6)

    # Weight by inverse distance (closer = stronger repulsion)
    weights = jnp.where(neighbors, 1.0 / (dists + 1e-6), 0.0)  # [n, n]

    # Weighted sum of directions away from neighbors
    force = jnp.sum(diff * weights[:, :, None], axis=1)  # [n, 2]

    # Normalize
    norm = jnp.linalg.norm(force, axis=-1, keepdims=True) + 1e-6
    return force / norm


def compute_alignment(
    vel: chex.Array,
    dists: chex.Array,
    perception_radius: float,
) -> chex.Array:
    """Compute alignment force: match velocity of neighbors.

    Args:
        vel: [n_agents, 2] velocities
        dists: [n_agents, n_agents] precomputed distances
        perception_radius: distance within which to consider neighbors

    Returns:
        [n_agents, 2] alignment forces (desired velocity direction)
    """
    neighbors = (dists < perception_radius) & (dists > 1e-6)
    n_neighbors = jnp.sum(neighbors, axis=1, keepdims=True) + 1e-6

    # Average velocity of neighbors
    avg_vel = jnp.sum(vel[None, :, :] * neighbors[:, :, None], axis=1) / n_neighbors

    # Normalize
    norm = jnp.linalg.norm(avg_vel, axis=-1, keepdims=True) + 1e-6
    return avg_vel / norm


def compute_cohesion(
    pos: chex.Array,
    dists: chex.Array,
    perception_radius: float,
) -> chex.Array:
    """Compute cohesion force: steer toward center of neighbors.

    Args:
        pos: [n_agents, 2] positions
        dists: [n_agents, n_agents] precomputed distances
        perception_radius: distance within which to consider neighbors

    Returns:
        [n_agents, 2] cohesion forces
    """
    neighbors = (dists < perception_radius) & (dists > 1e-6)
    n_neighbors = jnp.sum(neighbors, axis=1, keepdims=True) + 1e-6

    # Center of mass of neighbors
    center = jnp.sum(pos[None, :, :] * neighbors[:, :, None], axis=1) / n_neighbors

    # Direction toward center
    direction = center - pos

    # Normalize
    norm = jnp.linalg.norm(direction, axis=-1, keepdims=True) + 1e-6
    return direction / norm


def compute_boids_forces(
    pos: chex.Array,
    vel: chex.Array,
    perception_radius: float,
    separation_weight: float,
    alignment_weight: float,
    cohesion_weight: float,
) -> chex.Array:
    """Compute combined boids steering forces.

    Args:
        pos: [n_agents, 2] positions
        vel: [n_agents, 2] velocities
        perception_radius: neighbor detection radius
        separation_weight: weight for separation force
        alignment_weight: weight for alignment force
        cohesion_weight: weight for cohesion force

    Returns:
        [n_agents, 2] combined steering forces
    """
    dists = compute_distances(pos)

    sep = compute_separation(pos, dists, perception_radius)
    ali = compute_alignment(vel, dists, perception_radius)
    coh = compute_cohesion(pos, dists, perception_radius)

    return separation_weight * sep + alignment_weight * ali + cohesion_weight * coh


def compute_predator_attraction(
    predator_pos: chex.Array,
    prey_pos: chex.Array,
    prey_alive: chex.Array,
) -> chex.Array:
    """Compute attraction force toward nearest prey for each predator.

    Args:
        predator_pos: [n_pred, 2] predator positions
        prey_pos: [n_prey, 2] prey positions
        prey_alive: [n_prey] bool mask

    Returns:
        [n_pred, 2] attraction forces toward nearest alive prey
    """
    # Distance from each predator to each prey
    diff = prey_pos[None, :, :] - predator_pos[:, None, :]  # [n_pred, n_prey, 2]
    dists = jnp.linalg.norm(diff, axis=-1)  # [n_pred, n_prey]

    # Mask out dead prey with large distance
    dists = jnp.where(prey_alive[None, :], dists, 1e6)

    # Find nearest prey for each predator
    nearest_idx = jnp.argmin(dists, axis=1)  # [n_pred]

    # Direction to nearest prey
    direction = diff[jnp.arange(predator_pos.shape[0]), nearest_idx]  # [n_pred, 2]

    # Normalize
    norm = jnp.linalg.norm(direction, axis=-1, keepdims=True) + 1e-6
    return direction / norm


def compute_prey_avoidance(
    prey_pos: chex.Array,
    predator_pos: chex.Array,
    avoidance_radius: float = 20.0,
) -> chex.Array:
    """Compute avoidance force away from predators for each prey.

    Args:
        prey_pos: [n_prey, 2] prey positions
        predator_pos: [n_pred, 2] predator positions
        avoidance_radius: distance within which to flee

    Returns:
        [n_prey, 2] avoidance forces
    """
    # Direction from predators to prey
    diff = prey_pos[:, None, :] - predator_pos[None, :, :]  # [n_prey, n_pred, 2]
    dists = jnp.linalg.norm(diff, axis=-1)  # [n_prey, n_pred]

    # Weight by inverse distance (closer predators = more urgent)
    in_range = dists < avoidance_radius
    weights = jnp.where(in_range, 1.0 / (dists + 1e-6), 0.0)  # [n_prey, n_pred]

    # Weighted sum of escape directions
    force = jnp.sum(diff * weights[:, :, None], axis=1)  # [n_prey, 2]

    # Normalize (or zero if no predators nearby)
    norm = jnp.linalg.norm(force, axis=-1, keepdims=True) + 1e-6
    return force / norm


def wrap_positions(pos: chex.Array, world_size: float) -> chex.Array:
    """Wrap positions to stay within world boundaries (toroidal)."""
    return jnp.mod(pos, world_size)


def clip_velocity(vel: chex.Array, max_speed: float) -> chex.Array:
    """Clip velocity magnitude to max speed."""
    speed = jnp.linalg.norm(vel, axis=-1, keepdims=True)
    scale = jnp.minimum(1.0, max_speed / (speed + 1e-6))
    return vel * scale
