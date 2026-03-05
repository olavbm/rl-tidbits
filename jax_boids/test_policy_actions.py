"""Test that policy outputs are in reasonable range for learning."""

import jax
import jax.numpy as jnp

from jax_boids.ppo import create_train_state, make_distribution


def test_policy_action_magnitude():
    """Policy should output actions with reasonable magnitude, not all zeros."""
    key = jax.random.PRNGKey(42)
    train_state = create_train_state(key, 50, 2, 3e-4)

    # Test with various observation magnitudes
    key, k1 = jax.random.split(key)
    test_obs = [
        jnp.zeros(50),  # All zeros
        jnp.ones(50) * 0.5,  # Moderate values
        jax.random.normal(k1, (50,)),  # Random
    ]

    for obs in test_obs:
        out = train_state.apply_fn(train_state.params, obs)
        pi = make_distribution(out.action_mean, out.action_logstd)

        # Sample multiple actions to check distribution
        actions = jax.vmap(lambda k: pi.sample(seed=k))(jax.random.split(key, 100))

        # Actions should have non-zero std (not always outputting the same action)
        action_std = actions.std()
        print(
            f"Obs type: {obs.mean():.2f}, action std: {action_std:.4f}, mean: {actions.mean():.4f}"
        )

        # Std should be > 0.1 (reasonable exploration)
        assert action_std > 0.1, f"Action std too low: {action_std}"

        # Mean should be near zero (no bias)
        assert abs(actions.mean()) < 2.0, f"Action mean biased: {actions.mean()}"


def test_policy_action_range():
    """Policy actions should span a useful range for the environment."""
    train_state = create_train_state(jax.random.PRNGKey(0), 50, 2, 3e-4)

    # Sample many actions
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 1000)
    obs = jnp.zeros((1000, 50))

    out = jax.vmap(lambda o: train_state.apply_fn(train_state.params, o))(obs)

    # Sample from distribution
    actions = jax.vmap(lambda o, k: make_distribution(o[0], o[1]).sample(seed=k))(
        (out.action_mean, out.action_logstd), keys
    )

    print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"Action std: {actions.std():.3f}")

    # With logstd=0 (std=1), we expect ~99% of actions in [-3, 3]
    # Check that actions span a reasonable range
    assert actions.max() > 1.0, f"Actions too small: max={actions.max()}"
    assert actions.min() < -1.0, f"Actions too small: min={actions.min()}"


if __name__ == "__main__":
    test_policy_action_magnitude()
    print("\n---\n")
    test_policy_action_range()
