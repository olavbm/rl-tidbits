"""Sweep PPO hyperparameters for predator-prey.

Phase 2: Build on baseline success (1.91 prey_alive).
Test: longer training, partial LR anneal, param tuning.
"""

from jax_boids.envs.types import EnvConfig
from jax_boids.train_single import TrainConfig, train

CONFIGS = {
    "baseline_longer": TrainConfig(
        n_steps=128,
        n_epochs=4,
        ent_coef=0.01,
        n_envs=16,
        total_timesteps=5_000_000,
        prey_noise_scale=0.1,
        orthogonal_init=False,
        lr_anneal=False,
        min_lr=0.0,
        normalize_returns=False,
    ),
    "baseline_partial_anneal": TrainConfig(
        n_steps=128,
        n_epochs=4,
        ent_coef=0.01,
        n_envs=16,
        total_timesteps=1_000_000,
        prey_noise_scale=0.1,
        orthogonal_init=False,
        lr_anneal=True,
        min_lr=1e-4,
        normalize_returns=False,
    ),
    "baseline_clip_01": TrainConfig(
        n_steps=128,
        n_epochs=4,
        ent_coef=0.01,
        n_envs=16,
        total_timesteps=1_000_000,
        prey_noise_scale=0.1,
        clip_eps=0.1,
        orthogonal_init=False,
        lr_anneal=False,
        min_lr=0.0,
        normalize_returns=False,
    ),
    "baseline_clip_03": TrainConfig(
        n_steps=128,
        n_epochs=4,
        ent_coef=0.01,
        n_envs=16,
        total_timesteps=1_000_000,
        prey_noise_scale=0.1,
        clip_eps=0.3,
        orthogonal_init=False,
        lr_anneal=False,
        min_lr=0.0,
        normalize_returns=False,
    ),
    "baseline_lr_1e4": TrainConfig(
        n_steps=128,
        n_epochs=4,
        ent_coef=0.01,
        n_envs=16,
        total_timesteps=1_000_000,
        prey_noise_scale=0.1,
        lr=1e-4,
        orthogonal_init=False,
        lr_anneal=False,
        min_lr=0.0,
        normalize_returns=False,
    ),
    "baseline_lr_1e3": TrainConfig(
        n_steps=128,
        n_epochs=4,
        ent_coef=0.01,
        n_envs=16,
        total_timesteps=1_000_000,
        prey_noise_scale=0.1,
        lr=1e-3,
        orthogonal_init=False,
        lr_anneal=False,
        min_lr=0.0,
        normalize_returns=False,
    ),
}

ENV_CONFIG = EnvConfig(
    n_predators=5,
    n_prey=5,
    prey_learn=False,
    world_size=20.0,
)


def main():
    for name, config in CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"Running config: {name}")
        print(f"{'=' * 60}")
        print(f"  n_steps={config.n_steps}, n_epochs={config.n_epochs}, ent_coef={config.ent_coef}")
        print(
            f"  lr_anneal={config.lr_anneal}, ortho={config.orthogonal_init}, "
            f"norm_returns={config.normalize_returns}"
        )

        runner_state, metrics = train(
            config,
            ENV_CONFIG,
            seed=42,
            verbose=True,
            log_dir=f"runs/sweep/{name}",
        )

        final_prey = float(metrics["prey_alive"][-1])
        print(f"\n  {name} final prey_alive: {final_prey:.2f}")


if __name__ == "__main__":
    main()
