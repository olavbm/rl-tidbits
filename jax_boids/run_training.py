"""Quick training run with logging."""

from jax_boids.train import TrainConfig, EnvConfig, train

config = TrainConfig(
    total_timesteps=1_000_000,  # ~8000 updates
    n_envs=32,
    n_steps=128,
)
env_config = EnvConfig()

runner_state, metrics = train(
    config,
    env_config,
    seed=42,
    verbose=True,
    log_dir="runs",
)

print(f"\nTraining complete!")
print(f"  Pred policy_loss: {metrics['pred_policy_loss'][-1]:.4f}")
print(f"  Prey policy_loss: {metrics['prey_policy_loss'][-1]:.4f}")
print(f"  Final prey_alive: {metrics['prey_alive'][-1]:.1f}")
