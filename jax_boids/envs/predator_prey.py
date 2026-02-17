"""Predator-prey boids environment."""

from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp

from jax_boids.envs.boids import (
    clip_velocity,
    compute_boids_forces,
    compute_predator_attraction,
    compute_prey_avoidance,
    wrap_positions,
)
from jax_boids.envs.rewards import compute_predator_rewards, compute_prey_rewards
from jax_boids.envs.types import BoidsState, EnvConfig, obs_size


class PredatorPreyEnv:
    """JAX-based predator-prey boids environment.

    Predators try to catch prey, prey try to survive.
    Both teams exhibit flocking behavior via boids physics.
    """

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self._obs_size = obs_size(self.config)

    @property
    def observation_size(self) -> int:
        return self._obs_size

    @property
    def action_size(self) -> int:
        return 2  # 2D steering force

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], BoidsState]:
        """Initialize environment with random positions.

        Returns:
            observations: dict with 'predator' and 'prey' observation arrays
            state: initial BoidsState
        """
        cfg = self.config
        k1, k2, k3, k4, key = jax.random.split(key, 5)

        # Random initial positions
        predator_pos = jax.random.uniform(k1, (cfg.n_predators, 2)) * cfg.world_size
        prey_pos = jax.random.uniform(k2, (cfg.n_prey, 2)) * cfg.world_size

        # Random initial velocities (small)
        predator_vel = jax.random.uniform(k3, (cfg.n_predators, 2), minval=-1, maxval=1)
        prey_vel = jax.random.uniform(k4, (cfg.n_prey, 2), minval=-1, maxval=1)

        state = BoidsState(
            predator_pos=predator_pos,
            predator_vel=predator_vel,
            prey_pos=prey_pos,
            prey_vel=prey_vel,
            prey_alive=jnp.ones(cfg.n_prey, dtype=bool),
            step=0,
            key=key,
        )

        obs = self.get_obs(state)
        return obs, state

    def reset_from_state(self, state: BoidsState) -> Tuple[Dict[str, chex.Array], BoidsState]:
        """Initialize environment from a pre-constructed state.

        Useful for deterministic testing with known positions.

        Args:
            state: pre-constructed BoidsState

        Returns:
            observations: dict with 'predator' and 'prey' observation arrays
            state: the same state passed in
        """
        obs = self.get_obs(state)
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        state: BoidsState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[
        Dict[str, chex.Array], BoidsState, Dict[str, chex.Array], Dict[str, chex.Array], Dict
    ]:
        """Execute one environment step.

        Args:
            key: random key
            state: current state
            actions: dict with 'predator' [n_pred, 2] and 'prey' [n_prey, 2]

        Returns:
            observations, new_state, rewards, dones, info
        """
        cfg = self.config

        pred_actions = actions["predator"]  # [n_pred, 2]
        prey_actions = actions["prey"]  # [n_prey, 2]

        # Compute boids forces for prey only (predators hunt, don't flock)
        prey_boids = compute_boids_forces(
            state.prey_pos,
            state.prey_vel,
            cfg.perception_radius,
            cfg.separation_weight,
            cfg.alignment_weight,
            cfg.cohesion_weight,
        )

        # Add chase/evade instincts
        pred_chase = compute_predator_attraction(
            state.predator_pos, state.prey_pos, state.prey_alive
        )
        prey_flee = compute_prey_avoidance(state.prey_pos, state.predator_pos)

        # Combine forces: instinct + learned action (predators don't flock)
        pred_accel = pred_chase + pred_actions * cfg.max_acceleration
        prey_accel = prey_boids + prey_flee * 1.5 + prey_actions * cfg.max_acceleration

        # Update velocities
        new_pred_vel = state.predator_vel + pred_accel * cfg.dt
        new_prey_vel = state.prey_vel + prey_accel * cfg.dt

        # Apply speed limits (predators slightly faster)
        new_pred_vel = clip_velocity(new_pred_vel, cfg.max_speed * cfg.predator_speed_bonus)
        new_prey_vel = clip_velocity(new_prey_vel, cfg.max_speed)

        # Update positions
        new_pred_pos = state.predator_pos + new_pred_vel * cfg.dt
        new_prey_pos = state.prey_pos + new_prey_vel * cfg.dt

        # Wrap at boundaries
        new_pred_pos = wrap_positions(new_pred_pos, cfg.world_size)
        new_prey_pos = wrap_positions(new_prey_pos, cfg.world_size)

        # Check captures
        prey_alive, captures = self._check_captures(new_pred_pos, new_prey_pos, state.prey_alive)

        # Compute rewards
        pred_rewards, prey_rewards = self._compute_rewards(
            new_pred_pos, new_prey_pos, prey_alive, state.prey_alive, captures
        )

        # Update state
        new_state = state.replace(
            predator_pos=new_pred_pos,
            predator_vel=new_pred_vel,
            prey_pos=new_prey_pos,
            prey_vel=new_prey_vel,
            prey_alive=prey_alive,
            step=state.step + 1,
            key=key,
        )

        # Check done conditions
        all_prey_dead = ~jnp.any(prey_alive)
        max_steps_reached = new_state.step >= cfg.max_steps
        done = all_prey_dead | max_steps_reached

        obs = self.get_obs(new_state)

        rewards = {
            "predator": pred_rewards,
            "prey": prey_rewards,
        }
        dones = {
            "predator": jnp.full(cfg.n_predators, done),
            "prey": jnp.full(cfg.n_prey, done),
            "__all__": done,
        }
        info = {
            "prey_alive": jnp.sum(prey_alive),
            "captures_this_step": jnp.sum(captures),
        }

        return obs, new_state, rewards, dones, info

    def get_obs(self, state: BoidsState) -> Dict[str, chex.Array]:
        """Compute observations for all agents.

        Returns:
            dict with 'predator' [n_pred, obs_size] and 'prey' [n_prey, obs_size]
        """
        cfg = self.config

        pred_obs = jax.vmap(
            lambda i: self._single_agent_obs(
                i,
                state.predator_pos,
                state.predator_vel,
                state.prey_pos,
                state.prey_vel,
                n_same_team=cfg.n_predators,
                n_enemies=cfg.n_prey,
            )
        )(jnp.arange(cfg.n_predators))

        prey_obs = jax.vmap(
            lambda i: self._single_agent_obs(
                i,
                state.prey_pos,
                state.prey_vel,
                state.predator_pos,
                state.predator_vel,
                n_same_team=cfg.n_prey,
                n_enemies=cfg.n_predators,
            )
        )(jnp.arange(cfg.n_prey))

        return {"predator": pred_obs, "prey": prey_obs}

    def _single_agent_obs(
        self,
        agent_idx: int,
        own_team_pos: chex.Array,
        own_team_vel: chex.Array,
        enemy_pos: chex.Array,
        enemy_vel: chex.Array,
        n_same_team: int,
        n_enemies: int,
    ) -> chex.Array:
        """Compute observation for a single agent."""
        cfg = self.config

        my_pos = own_team_pos[agent_idx]
        my_vel = own_team_vel[agent_idx]

        # Distances to same team
        same_dists = jnp.linalg.norm(own_team_pos - my_pos, axis=-1)
        same_dists = same_dists.at[agent_idx].set(1e6)  # exclude self

        # K nearest same team (handle case where team is smaller than k)
        k_same = min(cfg.k_nearest_same, n_same_team - 1)
        _, same_idx = jax.lax.top_k(-same_dists, k_same)

        same_rel_pos_raw = own_team_pos[same_idx] - my_pos
        same_rel_vel_raw = own_team_vel[same_idx] - my_vel

        # Pad to fixed size if needed
        same_rel_pos = jnp.zeros((cfg.k_nearest_same, 2))
        same_rel_vel = jnp.zeros((cfg.k_nearest_same, 2))
        same_rel_pos = same_rel_pos.at[:k_same].set(same_rel_pos_raw)
        same_rel_vel = same_rel_vel.at[:k_same].set(same_rel_vel_raw)

        # Distances to enemies
        enemy_dists = jnp.linalg.norm(enemy_pos - my_pos, axis=-1)

        # K nearest enemies (handle case where enemies are fewer than k)
        k_enemy = min(cfg.k_nearest_enemy, n_enemies)
        _, enemy_idx = jax.lax.top_k(-enemy_dists, k_enemy)

        enemy_rel_pos_raw = enemy_pos[enemy_idx] - my_pos
        enemy_rel_vel_raw = enemy_vel[enemy_idx] - my_vel

        # Pad to fixed size if needed
        enemy_rel_pos = jnp.zeros((cfg.k_nearest_enemy, 2))
        enemy_rel_vel = jnp.zeros((cfg.k_nearest_enemy, 2))
        enemy_rel_pos = enemy_rel_pos.at[:k_enemy].set(enemy_rel_pos_raw)
        enemy_rel_vel = enemy_rel_vel.at[:k_enemy].set(enemy_rel_vel_raw)

        # Boundary distances
        boundary_dist = jnp.array(
            [
                my_pos[1],  # distance to bottom
                cfg.world_size - my_pos[1],  # distance to top
                my_pos[0],  # distance to left
                cfg.world_size - my_pos[0],  # distance to right
            ]
        )

        # Normalize observations
        same_rel_pos = same_rel_pos / cfg.world_size
        same_rel_vel = same_rel_vel / cfg.max_speed
        enemy_rel_pos = enemy_rel_pos / cfg.world_size
        enemy_rel_vel = enemy_rel_vel / cfg.max_speed
        boundary_dist = boundary_dist / cfg.world_size
        my_vel_norm = my_vel / cfg.max_speed

        return jnp.concatenate(
            [
                my_vel_norm,
                same_rel_pos.flatten(),
                same_rel_vel.flatten(),
                enemy_rel_pos.flatten(),
                enemy_rel_vel.flatten(),
                boundary_dist,
            ]
        )

    def _check_captures(
        self,
        pred_pos: chex.Array,
        prey_pos: chex.Array,
        prey_alive: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """Check which prey were captured.

        Returns:
            new_prey_alive: updated alive mask
            captures: [n_prey] bool mask of newly captured
        """
        cfg = self.config

        # Distance from each prey to each predator
        dists = jnp.linalg.norm(
            prey_pos[:, None, :] - pred_pos[None, :, :], axis=-1
        )  # [n_prey, n_pred]

        # Prey is captured if any predator is within capture radius
        min_dist = jnp.min(dists, axis=1)  # [n_prey]
        captured = (min_dist < cfg.capture_radius) & prey_alive
        new_prey_alive = prey_alive & ~captured

        return new_prey_alive, captured

    def _compute_rewards(
        self,
        pred_pos: chex.Array,
        prey_pos: chex.Array,
        prey_alive: chex.Array,
        prev_prey_alive: chex.Array,
        captures: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """Compute rewards for predators and prey."""
        n_captures = jnp.sum(captures)
        pred_rewards = compute_predator_rewards(n_captures, self.config.n_predators)
        prey_rewards = compute_prey_rewards(prey_pos, pred_pos, prey_alive, captures)
        return pred_rewards, prey_rewards
