"""Sanity check: verify predators can learn to chase stationary prey.

This test isolates the learning signal by:
1. Making prey completely stationary (no boids, no random motion)
2. Starting predators far from prey
3. Checking if learned policy improves over random baseline
"""

import chex
import jax
import jax.numpy as jnp

from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.types import BoidsState, EnvConfig


def test_reward_signal_exists():
    """Verify that moving toward prey yields positive reward gradient."""
    # Small config for speed
    env_config = EnvConfig(
        n_predators=2,
        n_prey=1,
        world_size=50.0,  # Small world
        max_steps=100,
        max_speed=5.0,
        max_acceleration=2.0,
        dt=0.1,
        separation_weight=0.0,  # No boids
        alignment_weight=0.0,
        cohesion_weight=0.0,
        perception_radius=15.0,
        capture_radius=3.0,
        predator_speed_bonus=1.2,
        k_nearest_same=1,
        k_nearest_enemy=1,
    )
    env = PredatorPreyEnv(env_config)

    # Deterministic initial state: prey at center, predators at corners
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4, key = jax.random.split(key, 5)

    # Prey stationary at center
    prey_pos = jnp.array([[25.0, 25.0]])
    prey_vel = jnp.zeros((1, 2))

    # Predators at opposite corners
    pred_pos = jnp.array([[5.0, 5.0], [45.0, 45.0]])
    pred_vel = jnp.zeros((2, 2))

    state = BoidsState(
        predator_pos=pred_pos,
        predator_vel=pred_vel,
        prey_pos=prey_pos,
        prey_vel=prey_vel,
        prey_alive=jnp.array([True]),
        step=0,
        key=key,
    )

    obs, state = env.reset_from_state(state)

    # Compute initial distances
    pred_to_prey = prey_pos[None, :] - pred_pos  # [2, 1, 2]
    distances = jnp.linalg.norm(pred_to_prey, axis=-1)
    print(f"Initial distances to prey: {distances}")
    print(f"Expected distance reward: {(50.0 - distances) * 0.05}")

    # Test 1: Random actions
    key, k1, k2, k3 = jax.random.split(key, 4)
    random_actions: dict[str, chex.Array] = {
        "predator": jax.random.uniform(k1, (2, 2), minval=-1, maxval=1),
        "prey": jnp.zeros((1, 2)),  # Stationary
    }
    _, _, random_rewards, _, _ = env.step(k2, state, random_actions)
    print(f"Random predator rewards: {random_rewards['predator']}")

    # Test 2: Direct chase actions
    chase_directions = pred_to_prey.mean(axis=1) / (
        jnp.linalg.norm(pred_to_prey.mean(axis=1), axis=-1, keepdims=True) + 1e-6
    )
    chase_actions: dict[str, chex.Array] = {
        "predator": chase_directions * 1.0,  # Steer toward prey
        "prey": jnp.zeros((1, 2)),
    }
    _, _, chase_rewards, _, _ = env.step(k3, state, chase_actions)
    print(f"Chase predator rewards: {chase_rewards['predator']}")

    # Compare
    print(f"Random avg: {random_rewards['predator'].mean():.4f}")
    print(f"Chase avg: {chase_rewards['predator'].mean():.4f}")

    # THE BUG: Both random and chase get similar rewards because the reward is based
    # on POSITION, not MOVEMENT TOWARD prey. The reward doesn't change based on
    # whether you're moving toward or away from prey - only on where you are.
    # This is a weak learning signal!

    # Let's trace through what happens:
    # 1. Initial: pred at [5,5], vel=[0,0]
    # 2. Action: chase direction * 1.0 = [0.707, 0.707] (normalized)
    # 3. Acceleration: action * max_acceleration = [0.707, 0.707] * 2.0 = [1.414, 1.414]
    # 4. New velocity: [0,0] + [1.414, 1.414] * 0.1 (dt) = [0.141, 0.141]
    # 5. New position: [5,5] + [0.141, 0.141] * 0.1 = [5.014, 5.014]
    # So in one step, predator moves only ~0.02 units! Very small.

    # This means the learning signal is extremely weak because:
    # 1. Predators move very slowly (physics constraints)
    # 2. Reward is based on distance, not progress toward prey
    # 3. One step of chasing barely changes distance


if __name__ == "__main__":
    test_reward_signal_exists()
