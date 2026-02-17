"""Neural network definitions for boids agents."""

from typing import NamedTuple, Sequence

import flax.linen as nn
import jax.numpy as jnp


class NetworkOutput(NamedTuple):
    """Output from actor-critic network."""

    action_mean: jnp.ndarray  # [batch?, action_dim]
    action_logstd: jnp.ndarray  # [action_dim] (shared across batch)
    value: jnp.ndarray  # [batch?]


class ActorCritic(nn.Module):
    """Simple MLP actor-critic network."""

    action_dim: int
    hidden_dims: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> NetworkOutput:
        # Shared layers
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.tanh(x)

        # Actor head
        action_mean = nn.Dense(self.action_dim)(x)
        action_logstd = self.param(
            "action_logstd",
            nn.initializers.zeros,
            (self.action_dim,),
        )

        # Critic head
        value = nn.Dense(1)(x)
        value = jnp.squeeze(value, axis=-1)

        return NetworkOutput(action_mean, action_logstd, value)
