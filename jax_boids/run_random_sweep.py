"""Large-scale random hyperparameter sweep.

Goal: Find optimal PPO hyperparameters across expanded search space.
Simple env: 1 predator, 3 prey, 10x10 world, 100 max steps.
"""

import argparse
import json
import random

from jax_boids.envs.types import EnvConfig
from jax_boids.train_single import TrainConfig, train

# Set seed for reproducibility
random.seed(42)

# Simple environment config
ENV_CONFIG = EnvConfig(
    n_predators=1,
    n_prey=3,
    prey_learn=False,
    world_size=10.0,
    max_steps=100,
)


def generate_configs(n_configs: int) -> dict:
    """Generate random configs."""
    configs = {}
    for i in range(n_configs):
        # Expanded search space
        lr = 10 ** random.uniform(-5, -1)  # 1e-5 to 1e-1
        clip = random.uniform(0.05, 0.5)  # 0.05 to 0.5
        ent = random.uniform(0.0, 0.2)  # 0.0 to 0.2
        vf_coef = random.uniform(0.25, 1.0)  # 0.25 to 1.0
        gae_lambda = random.uniform(0.85, 0.995)  # 0.85 to 0.995
        max_grad_norm = random.uniform(0.1, 1.0)  # 0.1 to 1.0

        # Discrete choices
        n_steps = random.choice([64, 128, 256, 512])
        n_epochs = random.choice([2, 4, 8, 10, 16])
        orthogonal_init = random.choice([True, False])
        lr_anneal = random.choice([True, False])
        normalize_returns = random.choice([True, False])
        min_lr_factor = random.choice([0.0, 0.1, 0.3])  # min_lr as fraction of initial LR

        configs[f"trial_{i:03d}"] = TrainConfig(
            lr=lr,
            clip_eps=clip,
            ent_coef=ent,
            vf_coef=vf_coef,
            gae_lambda=gae_lambda,
            max_grad_norm=max_grad_norm,
            n_steps=n_steps,
            n_epochs=n_epochs,
            n_envs=16,
            total_timesteps=200_000,
            prey_noise_scale=0.1,
            orthogonal_init=orthogonal_init,
            lr_anneal=lr_anneal,
            min_lr=min_lr_factor * lr if lr_anneal else 0.0,
            normalize_returns=normalize_returns,
        )
    return configs


def main():
    parser = argparse.ArgumentParser(description="Large-scale random hyperparameter sweep")
    parser.add_argument("--n-configs", type=int, default=100, help="Number of configs to test")
    args = parser.parse_args()

    # Generate configs
    CONFIGS = generate_configs(args.n_configs)

    # Load existing results if any
    existing_results = []
    try:
        with open("runs/expanded_random_sweep/results.json", "r") as f:
            existing_results = json.load(f)
        existing_names = {r["name"] for r in existing_results}
        print(f"Found {len(existing_results)} existing results")
    except FileNotFoundError:
        existing_names = set()

    # Filter to only run configs not yet completed
    configs_to_run = {name: cfg for name, cfg in CONFIGS.items() if name not in existing_names}
    print(f"Running {len(configs_to_run)} new configs out of {len(CONFIGS)} total")
    print()

    # Run each config
    new_results = []
    for name, config in configs_to_run.items():
        print(f"\n{'=' * 60}")
        print(f"Running config: {name}")
        print(f"{'=' * 60}")
        print(f"  lr={config.lr:.2e}, clip={config.clip_eps:.2f}, ent={config.ent_coef:.3f}")
        print(
            f"  n_steps={config.n_steps}, n_epochs={config.n_epochs}, vf_coef={config.vf_coef:.2f}"
        )

        runner_state, metrics = train(
            config,
            ENV_CONFIG,
            seed=42,
            verbose=True,
            log_dir=f"runs/expanded_random_sweep/{name}",
        )

        final_prey = float(metrics["prey_alive"][-1])
        final_kl = float(metrics["approx_kl"][-1]) if "approx_kl" in metrics else 0.0

        new_results.append(
            {
                "name": name,
                "lr": config.lr,
                "clip": config.clip_eps,
                "ent": config.ent_coef,
                "vf_coef": config.vf_coef,
                "gae_lambda": config.gae_lambda,
                "max_grad_norm": config.max_grad_norm,
                "n_steps": config.n_steps,
                "n_epochs": config.n_epochs,
                "orthogonal_init": config.orthogonal_init,
                "lr_anneal": config.lr_anneal,
                "min_lr": config.min_lr,
                "normalize_returns": config.normalize_returns,
                "prey_alive": final_prey,
                "kl": final_kl,
            }
        )

        print(f"\n  {name} final prey_alive: {final_prey:.2f}, KL: {final_kl:.4f}")
        print(
            f"    n_steps={config.n_steps}, n_epochs={config.n_epochs}, ortho={config.orthogonal_init}, lr_anneal={config.lr_anneal}"
        )

    # Combine with existing results
    all_results = existing_results + new_results

    # Sort by prey_alive (lower is better)
    all_results.sort(key=lambda x: x["prey_alive"])

    # Save configs
    config_summary = {}
    for name, cfg in locals()["CONFIGS"].items():
        config_summary[name] = {
            "lr": cfg.lr,
            "clip_eps": cfg.clip_eps,
            "ent_coef": cfg.ent_coef,
            "vf_coef": cfg.vf_coef,
            "gae_lambda": cfg.gae_lambda,
            "max_grad_norm": cfg.max_grad_norm,
            "n_steps": cfg.n_steps,
            "n_epochs": cfg.n_epochs,
            "orthogonal_init": cfg.orthogonal_init,
            "lr_anneal": cfg.lr_anneal,
            "min_lr": cfg.min_lr,
            "normalize_returns": cfg.normalize_returns,
            "total_timesteps": cfg.total_timesteps,
        }
    with open("runs/expanded_random_sweep/configs.json", "w") as f:
        json.dump(config_summary, f, indent=2)

    # Save results
    with open("runs/expanded_random_sweep/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("TOP 50 RESULTS (sorted by prey_alive)")
    print("=" * 60)
    print(
        f"{'Rank':<6}{'Name':<12}{'Prey':<8}{'KL':<10}{'LR':<12}{'Clip':<8}{'Ent':<8}{'n_steps':<8}{'n_epochs':<6}"
    )
    print("-" * 80)
    for rank, r in enumerate(all_results[:50], 1):
        print(
            f"{rank:<6}{r['name']:<12}{r['prey_alive']:<8.2f}"
            f"{r['kl']:<10.4f}{r['lr']:<12.2e}{r['clip']:<8.2f}"
            f"{r['ent']:<8.3f}{r['n_steps']:<8}{r['n_epochs']:<6}"
        )

    print("\nResults saved to runs/expanded_random_sweep/results.json")


if __name__ == "__main__":
    main()
