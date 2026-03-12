"""Run incremental training experiments with logging."""

from jax_boids.train import TrainConfig, EnvConfig, train

# Incremental training runs
configs = [
    TrainConfig(total_timesteps=2_000_000, n_envs=32, n_steps=128),  # ~16000 updates
]

for i, config in enumerate(configs):
    print(f"\n{'=' * 60}")
    print(f"Run {i + 1}: {config.total_timesteps:,} timesteps")
    print(f"{'=' * 60}")

    env_config = EnvConfig()
    runner_state, metrics = train(
        config,
        env_config,
        seed=42,
        verbose=True,
        log_dir="runs",
    )

    print(f"\nRun {i + 1} complete!")
    print(f"  Pred policy_loss: {metrics['pred_policy_loss'][-1]:.4f}")
    print(f"  Prey policy_loss: {metrics['prey_policy_loss'][-1]:.4f}")
    print(f"  Final prey_alive: {metrics['prey_alive'][-1]:.1f}")
