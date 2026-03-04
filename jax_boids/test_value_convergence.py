"""Test if value function can learn to predict fixed targets with continuous training."""

import jax
import jax.numpy as jnp
from jax_boids.networks import ActorCritic
from jax_boids.ppo import compute_gae, create_train_state, ppo_loss

key = jax.random.PRNGKey(42)

# Create fixed dummy data
n_steps = 2048
n_agents = 4
obs_size = 60
action_size = 2

# Fixed observations [T, N, obs_size] - add timestep encoding to see if network can learn varying values
time_encoding = jnp.linspace(0, 1, n_steps)[:, None, None]  # [T, 1, 1]
obs = jnp.ones((n_steps, n_agents, obs_size)) * 0.5
obs = obs + time_encoding * 0.1  # Add small timestep signal

# Fixed rewards (constant positive reward)
rewards = jnp.ones((n_agents, n_steps)) * 1.0

# Fixed dones (all zeros except last)
dones = jnp.zeros((n_agents, n_steps))
dones = dones.at[:, -1].set(1.0)

# Create dummy actions and log_probs for PPO loss function
actions = jnp.zeros((n_steps, n_agents, action_size))
old_log_probs = jnp.zeros((n_steps, n_agents))

# Create train state
train_state = create_train_state(key, obs_size, action_size, lr=0.001)

# Compute returns
gamma = 0.99
lambd = 0.95


# Get initial value estimates
def get_values(train_state, obs):
    # obs: [T, N, obs_size] - vmap over first two dims
    obs_flat = obs.reshape(-1, obs.shape[-1])
    out = jax.vmap(lambda o: train_state.apply_fn(train_state.params, o))(obs_flat)
    values = out.value
    return values.reshape(obs.shape[:2])


# Compute GAE targets
# compute_gae expects [T, N] format
values = get_values(train_state, obs)  # [T, N]
values_next = get_values(train_state, obs[-1:, :, :])  # [1, N]
values_with_bootstrap = jnp.concatenate([values, values_next], axis=0)  # [T+1, N]
rewards_T = rewards.T  # [T, N]
dones_T = dones.T  # [T, N]
gae, returns = compute_gae(rewards_T, values_with_bootstrap, dones_T, gamma, lambd)

# Flatten for ppo_loss: [T*N, ...]
obs_flat = obs.reshape(-1, obs_size)  # [T*N, obs_size]
actions_flat = actions.reshape(-1, action_size)  # [T*N, action_size]
old_log_probs_flat = old_log_probs.flatten()  # [T*N]
gae_flat = gae.flatten()  # [T*N]
returns_flat = returns.flatten()  # [T*N]

print("=== Fixed Data Statistics ===")
print(f"Returns mean: {returns.mean():.4f}")
print(f"Returns std: {returns.std():.4f}")
print(f"Returns min: {returns.min():.4f}")
print(f"Returns max: {returns.max():.4f}")
print(f"Initial value mean: {values.mean():.4f}")
print(f"Initial value std: {values.std():.4f}")


# Train value function continuously on this fixed data
def train_step(train_state, key):
    def loss_fn(params):
        _, metrics = ppo_loss(
            params,
            lambda p, x: train_state.apply_fn(p, x),
            obs_flat,
            actions_flat,
            old_log_probs_flat,
            gae_flat,
            returns_flat,
            clip_eps=0.2,
            vf_coef=1.0,
            ent_coef=0.0,
        )
        return metrics["value_loss"]

    grad = jax.grad(loss_fn)(train_state.params)
    new_state = train_state.apply_gradients(grads=grad)
    loss_val = loss_fn(train_state.params)
    return new_state, loss_val


# Training loop
print("\n=== Training Value Function on Fixed Data ===")
for epoch in range(500):
    train_state, v_loss = train_step(train_state, key)

    # Evaluate current value predictions
    current_values = get_values(train_state, obs)
    error = jnp.mean((current_values - returns) ** 2)

    if epoch % 50 == 0:
        print(
            f"Epoch {epoch:4d}: "
            f"Value Loss={v_loss:.4f}, "
            f"MSE={error:.4f}, "
            f"Value Mean={current_values.mean():.4f}, "
            f"Value Std={current_values.std():.4f}"
        )

# Final evaluation
final_values = get_values(train_state, obs)
print("\n=== Final Evaluation ===")
print(f"Target (returns) mean: {returns.mean():.4f}")
print(f"Target (returns) std: {returns.std():.4f}")
print(f"Predicted values mean: {final_values.mean():.4f}")
print(f"Predicted values std: {final_values.std():.4f}")
print(f"Final MSE: {jnp.mean((final_values - returns) ** 2):.4f}")
print(f"Final MAE: {jnp.mean(jnp.abs(final_values - returns)):.4f}")

# Check if learned correctly
final_mse = jnp.mean((final_values - returns) ** 2)
returns_var = jnp.var(returns)
print(f"\nReturns variance: {returns_var:.4f}")
print(f"Final MSE: {final_mse:.4f}")

# The MSE should approach the variance if the network predicts the mean perfectly
# (since MSE = variance + (bias)^2, and bias should be ~0)
if final_mse < returns_var * 1.5:  # Allow 50% tolerance
    print("\n✓ Value function successfully learned to predict the mean!")
    print("  (MSE ≈ variance, meaning the network predicts the correct mean)")
else:
    print("\n✗ Value function failed to learn fixed targets")
