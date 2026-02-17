"""State and configuration types for boids environments."""

from typing import NamedTuple

import chex
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class BoidsState:
    """Immutable state for predator-prey boids environment."""

    predator_pos: chex.Array  # [n_predators, 2]
    predator_vel: chex.Array  # [n_predators, 2]
    prey_pos: chex.Array  # [n_prey, 2]
    prey_vel: chex.Array  # [n_prey, 2]
    prey_alive: chex.Array  # [n_prey] bool mask
    step: int
    key: chex.PRNGKey


@struct.dataclass
class EnvConfig:
    """Configuration for the predator-prey environment."""

    # Agent counts
    n_predators: int = 5
    n_prey: int = 10

    # World parameters
    world_size: float = 100.0
    max_steps: int = 500

    # Physics
    max_speed: float = 5.0
    max_acceleration: float = 2.0
    dt: float = 0.1

    # Boids parameters
    separation_weight: float = 1.5
    alignment_weight: float = 1.0
    cohesion_weight: float = 1.0
    perception_radius: float = 15.0

    # Predator-prey interaction
    capture_radius: float = 2.0
    predator_speed_bonus: float = 1.2  # predators slightly faster

    # Observation parameters
    k_nearest_same: int = 4  # observe 4 nearest same-team
    k_nearest_enemy: int = 3  # observe 3 nearest enemy


class Observation(NamedTuple):
    """Observation structure for a single agent."""

    velocity: chex.Array  # [2] own velocity
    same_team_rel_pos: chex.Array  # [k_same, 2] relative positions
    same_team_rel_vel: chex.Array  # [k_same, 2] relative velocities
    enemy_rel_pos: chex.Array  # [k_enemy, 2] relative positions
    enemy_rel_vel: chex.Array  # [k_enemy, 2] relative velocities
    boundary_dist: chex.Array  # [4] distance to boundaries (up, down, left, right)


def obs_size(config: EnvConfig) -> int:
    """Calculate total observation vector size."""
    return (
        2  # own velocity
        + config.k_nearest_same * 2  # same team relative positions
        + config.k_nearest_same * 2  # same team relative velocities
        + config.k_nearest_enemy * 2  # enemy relative positions
        + config.k_nearest_enemy * 2  # enemy relative velocities
        + 4  # boundary distances
    )


def flatten_obs(obs: Observation) -> chex.Array:
    """Flatten observation to single vector."""
    return jnp.concatenate(
        [
            obs.velocity,
            obs.same_team_rel_pos.flatten(),
            obs.same_team_rel_vel.flatten(),
            obs.enemy_rel_pos.flatten(),
            obs.enemy_rel_vel.flatten(),
            obs.boundary_dist,
        ]
    )
