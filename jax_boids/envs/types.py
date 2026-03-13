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
class CurriculumStage:
    """Configuration for a single curriculum stage."""

    name: str
    n_prey: int
    world_size: float
    prey_speed_mult: float = 1.0
    max_steps: int = 500
    predator_speed_mult: float = 1.0


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
    max_speed: float = 1.0
    max_acceleration: float = 0.5
    dt: float = 0.1
    velocity_damping: float = 0.9  # damping factor (1.0 = no damping)

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

    # Learning mode
    prey_learn: bool = True  # if False, prey don't learn (passive boids)
    distance_reward: bool = True  # if False, only capture reward for predators
    prey_speed_mult: float = 1.0  # multiplier for prey max speed (curriculum)

    # Curriculum
    curriculum: list[CurriculumStage] | None = None
    curriculum_timesteps: int = 0
    current_stage: int = 0


class Observation(NamedTuple):
    """Observation structure for a single agent."""

    velocity: chex.Array  # [2] own velocity
    same_team_rel_pos: chex.Array  # [k_same, 2] relative positions
    same_team_rel_vel: chex.Array  # [k_same, 2] relative velocities
    enemy_rel_pos: chex.Array  # [k_enemy, 2] relative positions
    enemy_rel_vel: chex.Array  # [k_enemy, 2] relative velocities


def obs_size(config: EnvConfig) -> int:
    """Calculate total observation vector size."""
    return (
        2  # own velocity
        + config.k_nearest_same * 2  # same team relative positions
        + config.k_nearest_same * 2  # same team relative velocities
        + config.k_nearest_enemy * 2  # enemy relative positions
        + config.k_nearest_enemy * 2  # enemy relative velocities
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
        ]
    )
