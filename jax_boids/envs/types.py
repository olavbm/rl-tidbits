"""State and configuration types for boids environments."""

from typing import NamedTuple, Optional

import chex
import jax.numpy as jnp
from flax import struct


class TrainConfig(NamedTuple):
    """Training hyperparameters for PPO training.

    Prey-specific overrides (prey_*) default to None, meaning use the
    predator value. Only used in IPPO training where both sides learn.
    """

    # Predator (and default) hyperparameters
    lr: float = 0.0027
    gamma: float = 0.99
    gae_lambda: float = 0.87
    clip_eps: float = 0.33
    vf_coef: float = 0.93
    ent_coef: float = 0.048
    max_grad_norm: float = 0.515
    n_steps: int = 256
    n_epochs: int = 10
    n_minibatches: int = 4
    total_timesteps: int = 1_000_000
    n_envs: int = 32
    prey_noise_scale: float = 0.3
    orthogonal_init: bool = False
    lr_anneal: bool = False
    min_lr: float = 0.0
    normalize_returns: bool = True
    log_interval: int = 50
    checkpoint_interval: int = 500

    # Prey-specific overrides (None = use predator value)
    prey_lr: Optional[float] = None
    prey_gamma: Optional[float] = None
    prey_gae_lambda: Optional[float] = None
    prey_clip_eps: Optional[float] = None
    prey_vf_coef: Optional[float] = None
    prey_ent_coef: Optional[float] = None
    prey_max_grad_norm: Optional[float] = None
    prey_orthogonal_init: Optional[bool] = None
    prey_lr_anneal: Optional[bool] = None
    prey_min_lr: Optional[float] = None
    prey_normalize_returns: Optional[bool] = None


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
    n_predators: int = 1
    n_prey: int = 3

    # World parameters
    world_size: float = 10.0
    max_steps: int = 200

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
    capture_radius: float = 0.3
    predator_speed_bonus: float = 1.2  # predators slightly faster

    # Observation parameters
    k_nearest_same: int = 4  # observe 4 nearest same-team
    k_nearest_enemy: int = 3  # observe 3 nearest enemy

    # Learning mode
    prey_learn: bool = False  # if False, prey don't learn (passive boids)
    distance_reward: bool = True  # if False, only capture reward for predators
    prey_speed_mult: float = 0.5  # multiplier for prey max speed
    boids_strength: float = 1.0  # multiplier for all boids forces (< 1.0 weakens flocking)


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
        + 2  # own position (normalized)
        + 1  # agent index (normalized)
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
