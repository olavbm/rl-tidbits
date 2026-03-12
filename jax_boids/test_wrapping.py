"""Tests for toroidal wrapping correctness."""

import jax
import jax.numpy as jnp

from jax_boids.envs.boids import wrapped_diff
from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.types import BoidsState, EnvConfig


def _make_state(pred_pos, prey_pos, world_size=20.0, **kwargs):
    pred_pos = jnp.array(pred_pos, dtype=jnp.float32)
    prey_pos = jnp.array(prey_pos, dtype=jnp.float32)
    n_pred = pred_pos.shape[0]
    n_prey = prey_pos.shape[0]
    return BoidsState(
        predator_pos=pred_pos,
        predator_vel=kwargs.get("pred_vel", jnp.zeros((n_pred, 2))),
        prey_pos=prey_pos,
        prey_vel=kwargs.get("prey_vel", jnp.zeros((n_prey, 2))),
        prey_alive=jnp.array(kwargs.get("prey_alive", [True] * n_prey)),
        step=0,
        key=jax.random.PRNGKey(0),
    )


def test_wrapped_diff_basic():
    """Shortest path in toroidal world."""
    ws = 20.0
    # Same side: normal subtraction
    a = jnp.array([5.0, 5.0])
    b = jnp.array([3.0, 3.0])
    d = wrapped_diff(a, b, ws)
    assert jnp.allclose(d, jnp.array([2.0, 2.0])), f"Got {d}"

    # Across boundary: should wrap
    a = jnp.array([1.0, 1.0])
    b = jnp.array([19.0, 19.0])
    d = wrapped_diff(a, b, ws)
    assert jnp.allclose(d, jnp.array([2.0, 2.0])), f"Expected [2,2], got {d}"

    # Other direction
    a = jnp.array([19.0, 19.0])
    b = jnp.array([1.0, 1.0])
    d = wrapped_diff(a, b, ws)
    assert jnp.allclose(d, jnp.array([-2.0, -2.0])), f"Expected [-2,-2], got {d}"


def test_capture_across_boundary():
    """Predator at pos 19 should capture prey at pos 1 in a 20x20 world."""
    cfg = EnvConfig(
        n_predators=1, n_prey=1, world_size=20.0, capture_radius=3.0,
        k_nearest_same=0, k_nearest_enemy=1, max_steps=100,
    )
    env = PredatorPreyEnv(cfg)

    # Predator at (19, 10), prey at (1, 10) — wrapped distance = 2, within capture_radius
    state = _make_state(
        pred_pos=[[19.0, 10.0]],
        prey_pos=[[1.0, 10.0]],
        world_size=20.0,
    )
    key = jax.random.PRNGKey(0)
    actions = {"predator": jnp.zeros((1, 2)), "prey": jnp.zeros((1, 2))}
    _, _, _, _, info = env.step(key, state, actions)
    assert int(info["captures_this_step"]) == 1, (
        f"Should capture across boundary, got {info['captures_this_step']}"
    )


def test_no_capture_without_wrapping_would_miss():
    """Agents far apart in raw coords but close when wrapped should still capture."""
    cfg = EnvConfig(
        n_predators=1, n_prey=1, world_size=20.0, capture_radius=3.0,
        k_nearest_same=0, k_nearest_enemy=1, max_steps=100,
    )
    env = PredatorPreyEnv(cfg)

    # Predator at (18.5, 10), prey at (0.5, 10) — wrapped distance = 2
    state = _make_state(
        pred_pos=[[18.5, 10.0]],
        prey_pos=[[0.5, 10.0]],
        world_size=20.0,
    )
    key = jax.random.PRNGKey(0)
    actions = {"predator": jnp.zeros((1, 2)), "prey": jnp.zeros((1, 2))}
    _, _, _, _, info = env.step(key, state, actions)
    assert int(info["captures_this_step"]) == 1, (
        f"Wrapped distance is 2, should capture. Got {info['captures_this_step']}"
    )


def test_dead_prey_not_in_obs():
    """Dead prey should appear as zeros in predator observations."""
    cfg = EnvConfig(
        n_predators=1, n_prey=2, world_size=100.0,
        k_nearest_same=0, k_nearest_enemy=2, max_steps=100,
    )
    env = PredatorPreyEnv(cfg)

    state = _make_state(
        pred_pos=[[50.0, 50.0]],
        prey_pos=[[55.0, 50.0], [60.0, 50.0]],
        prey_alive=[True, False],
        world_size=100.0,
    )
    obs = env.get_obs(state)
    pred_obs = obs["predator"][0]

    # obs layout: [my_vel(2), same_rel_pos(0*2), same_rel_vel(0*2), enemy_rel_pos(2*2), enemy_rel_vel(2*2)]
    # With k_nearest_same=0: no same-team obs
    # enemy_rel_pos starts at index 2
    enemy_rel_pos = pred_obs[2:6].reshape(2, 2)  # 2 enemies, 2D each

    # First enemy should be alive prey (at 55,50), second should be zeros (dead)
    half_world = cfg.world_size / 2.0
    expected_alive = jnp.array([5.0 / half_world, 0.0])  # relative pos to alive prey
    assert jnp.allclose(enemy_rel_pos[0], expected_alive, atol=1e-4), (
        f"Expected alive prey obs {expected_alive}, got {enemy_rel_pos[0]}"
    )
    assert jnp.allclose(enemy_rel_pos[1], jnp.zeros(2), atol=1e-4), (
        f"Expected dead prey obs zeros, got {enemy_rel_pos[1]}"
    )


def test_obs_wrapping():
    """Observations should show shortest path across boundary."""
    cfg = EnvConfig(
        n_predators=1, n_prey=1, world_size=20.0,
        k_nearest_same=0, k_nearest_enemy=1, max_steps=100,
    )
    env = PredatorPreyEnv(cfg)

    # Predator at (19, 10), prey at (1, 10) — wrapped relative pos should be (2, 0)
    state = _make_state(
        pred_pos=[[19.0, 10.0]],
        prey_pos=[[1.0, 10.0]],
        world_size=20.0,
    )
    obs = env.get_obs(state)
    pred_obs = obs["predator"][0]

    # enemy_rel_pos starts at index 2 (after my_vel)
    half_world = cfg.world_size / 2.0
    expected_x = 2.0 / half_world  # wrapped distance
    # The x component should be small and positive (prey is 2 units ahead via wrapping)
    enemy_rel_x = pred_obs[2]
    assert jnp.allclose(enemy_rel_x, expected_x, atol=1e-4), (
        f"Expected wrapped rel_x={expected_x}, got {enemy_rel_x}"
    )


if __name__ == "__main__":
    test_wrapped_diff_basic()
    print("OK: wrapped_diff basic")
    test_capture_across_boundary()
    print("OK: capture across boundary")
    test_no_capture_without_wrapping_would_miss()
    print("OK: capture with wrapping")
    test_dead_prey_not_in_obs()
    print("OK: dead prey masked in obs")
    test_obs_wrapping()
    print("OK: obs wrapping")
    print("\nAll wrapping tests passed!")
