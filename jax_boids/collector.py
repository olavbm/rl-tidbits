"""Rollout collection utilities for boids environments.

Separates environment interaction from training logic.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict

import chex
import jax
import jax.numpy as jnp

from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.types import EnvConfig
from jax_boids.networks import NetworkOutput
from jax_boids.ppo import (
    Transition,
    make_distribution,
    select_on_reset,
)
from flax.training.train_state import TrainState


class PolicyType(Enum):
    """Type of policy for an agent."""

    LEARNED = "learned"
    RANDOM = "random"


@dataclass
class PolicyConfig:
    """Configure a policy for a given agent type.

    Attributes:
        policy_type: Whether agent uses learned policy or random actions
        train_state: Training state for learned policies (LEARNED only)
        noise_scale: Scale factor for random actions (RANDOM only)
    """

    policy_type: PolicyType
    train_state: TrainState | None = None
    noise_scale: float = 0.3


@dataclass
class RolloutConfig:
    """Configuration for rollout collection.

    Attributes:
        n_steps: Number of environment steps per rollout
        n_envs: Number of parallel environments
    """

    n_steps: int = 128
    n_envs: int = 32


PolicyFunction = Callable[[chex.Array, TrainState], NetworkOutput]


def create_policy_fn(
    policy_config: PolicyConfig,
) -> Callable[[chex.Array, chex.PRNGKey], tuple[chex.Array, chex.Array]]:
    """Create forward function for policy that returns (actions, log_probs).

    Handles both learned policies and random action generation.

    Args:
        policy_config: Configuration specifying policy type and parameters

    Returns:
        Function that takes (obs, key) and returns (actions, log_probs)
        where actions and log_probs have shapes [n_agents, action_size] and [n_agents]
    """

    def learned_policy(obs: chex.Array, key: chex.PRNGKey) -> tuple[chex.Array, chex.Array]:
        """Apply learned policy: obs → network → sample from distribution."""
        if policy_config.train_state is None:
            raise ValueError("train_state required for LEARNED policy type")

        out = jax.vmap(
            lambda o: policy_config.train_state.apply_fn(policy_config.train_state.params, o)
        )(obs)
        pi = make_distribution(out.action_mean, out.action_logstd)
        actions = pi.sample(seed=key)
        log_probs = pi.log_prob(actions)
        return actions, log_probs

    def random_policy(obs: chex.Array, key: chex.PRNGKey) -> tuple[chex.Array, chex.Array]:
        """Generate random actions with optional noise scaling."""
        n_agents = obs.shape[0]
        action_size = obs.shape[1] // (obs.shape[1] // 2)

        scale = policy_config.noise_scale
        actions = jax.random.uniform(key, (n_agents, 2), minval=-scale, maxval=scale)
        log_probs = jnp.zeros((n_agents,))
        return actions, log_probs

    if policy_config.policy_type == PolicyType.LEARNED:
        return learned_policy
    elif policy_config.policy_type == PolicyType.RANDOM:
        return random_policy
    else:
        raise ValueError(f"Unknown policy type: {policy_config.policy_type}")


def collect_rollouts(
    env: PredatorPreyEnv,
    policies: Dict[str, PolicyConfig],
    env_config: EnvConfig,
    rollout_config: RolloutConfig,
    key: chex.PRNGKey,
    obs: Dict[str, chex.Array] | None = None,
    env_state: chex.Array | None = None,
) -> tuple[
    chex.PRNGKey,
    tuple[Dict[str, Transition], Dict[str, chex.Array]],
    Dict[str, chex.Array],
    chex.Array,
]:
    """Collect rollout transitions for all agent types.

    Runs the environment for n_steps, collecting transitions for each agent type.
    Each agent type can have its own policy configuration (learned or random).

    Args:
        env: The predator-prey environment
        policies: Dict mapping agent type ("predator", "prey") → policy config
        env_config: Environment configuration
        rollout_config: Rollout hyperparameters
        key: PRNG key
        obs: Initial observations for each agent type. If None, calls env.reset()
        env_state: Initial environment state. If None, calls env.reset()

    Returns:
        Updated key
        Tuple of (transitions dict, info dict)
        Final observations for each agent type
        Final environment state
    """
    n_pred = env_config.n_predators
    n_prey = env_config.n_prey
    n_envs = rollout_config.n_envs
    n_steps = rollout_config.n_steps

    # Initialize from reset if no state provided
    if obs is None or env_state is None:
        env_keys = jax.random.split(key, n_envs)
        obs, env_state = jax.vmap(env.reset)(env_keys)

    # Create policy functions for each agent type
    policy_fns = {agent_type: create_policy_fn(config) for agent_type, config in policies.items()}

    def _env_step(carry, _):
        """Single environment step for rollout collection.

        Args:
            carry: (key, obs, env_state)
            _: Unused (for lax.scan)

        Returns:
            Updated carry
            Tuple of transitions (one per agent type)
        """
        key, obs, env_state = carry

        agent_keys = jax.random.split(key, len(policies) + 1)
        keys = list(agent_keys[:-1])
        step_key = agent_keys[-1]

        # Get actions and log probs for each agent type
        actions = {}
        log_probs = {}
        values = {}

        for agent_type, policy_fn in policy_fns.items():
            n_agents = n_pred if agent_type == "predator" else n_prey
            agent_obs = obs[agent_type]  # [n_envs, n_agents, obs_size]

            # Flatten for policy: [n_envs, n_agents, obs_size] → [n_envs * n_agents, obs_size]
            agent_obs_flat = agent_obs.reshape(-1, agent_obs.shape[-1])
            key, k = jax.random.split(key)

            actions_flat, log_probs_flat = policy_fn(agent_obs_flat, k)

            # Reshape back: [n_envs * n_agents, ...] → [n_envs, n_agents, ...]
            actions[agent_type] = actions_flat.reshape(n_envs, n_agents, -1)
            log_probs[agent_type] = log_probs_flat.reshape(n_envs, n_agents)

            # Only compute values for LEARNED policies
            if policies[agent_type].policy_type == PolicyType.LEARNED:
                train_state = policies[agent_type].train_state
                out = jax.vmap(lambda o: train_state.apply_fn(train_state.params, o))(
                    agent_obs_flat
                )
                values[agent_type] = out.value.reshape(n_envs, n_agents)

        # Environment step
        step_keys = jax.random.split(step_key, n_envs)
        next_obs, env_state_new, rewards, dones, info = jax.vmap(env.step)(
            step_keys, env_state, actions
        )

        # Build transitions for each agent type
        transitions = {}
        for agent_type in policies.keys():
            transitions[agent_type] = Transition(
                obs=obs[agent_type],
                action=actions[agent_type],
                reward=rewards[agent_type],
                done=dones[agent_type],
                log_prob=log_probs[agent_type],
                value=values.get(
                    agent_type, jnp.zeros((n_envs, n_pred if agent_type == "predator" else n_prey))
                ),
            )

        # Handle resets
        reset_mask = dones["__all__"]
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, n_envs)
        new_obs, new_states = jax.vmap(env.reset)(reset_keys)

        def _select(new, old):
            return select_on_reset(reset_mask, new, old)

        new_obs_dict = jax.tree.map(_select, new_obs, next_obs)
        new_env_state = jax.tree.map(_select, new_states, env_state_new)

        return (key, new_obs_dict, new_env_state), (transitions, {k: v for k, v in info.items()})

    # Roll out for n_steps
    (key, final_obs, final_env_state), (all_transitions, all_infos) = jax.lax.scan(
        _env_step, (key, obs, env_state), None, length=n_steps
    )

    return key, (all_transitions, all_infos), final_obs, final_env_state
