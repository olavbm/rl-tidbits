"""Fine-tuning sweep around best config from expanded random sweep.

Best config (trial_099):
  lr=1.3e-3, clip=0.219, ent=0.091, vf_coef=0.368, gae_lambda=0.895
  max_grad_norm=0.392, n_steps=64, n_epochs=2
  ortho=True, lr_anneal=True, min_lr=0.3x, norm_ret=True
  prey_alive=1.415

This script runs a targeted sweep with tighter bounds around these values.
"""

import argparse
import json
import random
from pathlib import Path

from jax_boids.envs.types import EnvConfig
from jax_boids.train_single import TrainConfig, train

# Set seed for reproducibility
random.seed(42)

# Simple environment config (same as expanded sweep)
ENV_CONFIG = EnvConfig(
    n_predators=1,
    n_prey=3,
    prey_learn=False,
    world_size=10.0,
    max_steps=100,
)


def generate_configs(n_configs: int) -> dict:
    """Generate configs with tighter bounds around trial_099."""
    configs = {}

    # Tighter bounds around best config
    # Trial_099: lr=1.3e-3, clip=0.219, ent=0.091, vf=0.368, gae=0.895, grad=0.392
    LR_MIN, LR_MAX = -3.0, -2.7  # log10 bounds for 1e-3 to 2e-3 (narrower around 1.3e-3)
    CLIP_MIN, CLIP_MAX = 0.15, 0.30  # narrower around 0.22
    ENT_MIN, ENT_MAX = 0.05, 0.15  # narrower around 0.09
    VF_MIN, VF_MAX = 0.25, 0.50  # narrower around 0.37
    GAE_MIN, GAE_MAX = 0.85, 0.95  # narrower around 0.90
    GRAD_MIN, GRAD_MAX = 0.25, 0.50  # narrower around 0.39

    for i in range(n_configs):
        lr = 10 ** random.uniform(LR_MIN, LR_MAX)
        clip = random.uniform(CLIP_MIN, CLIP_MAX)
        ent = random.uniform(ENT_MIN, ENT_MAX)
        vf_coef = random.uniform(VF_MIN, VF_MAX)
        gae_lambda = random.uniform(GAE_MIN, GAE_MAX)
        max_grad_norm = random.uniform(GRAD_MIN, GRAD_MAX)

        # Discrete choices - focus on n_steps=64, n_epochs=2 but allow some variation
        n_steps = random.choice([32, 64, 64, 64, 128])  # weight towards 64
        n_epochs = random.choice([2, 2, 2, 4, 4])  # weight towards 2

        # Always enable the three features that worked well
        orthogonal_init = True
        lr_anneal = True
        normalize_returns = True

        # Min LR: focus on 0.3x (worked well in trial_099)
        min_lr_factor = random.choice([0.1, 0.2, 0.3, 0.3, 0.3])
        min_lr = min_lr_factor * lr

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
            min_lr=min_lr,
            normalize_returns=normalize_returns,
        )

    return configs


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning sweep around best config")
    parser.add_argument("--n-configs", type=int, default=100, help="Number of configs to test")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/fine_tuning_sweep",
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate configs
    configs = generate_configs(args.n_configs)

    # Load existing results if any
    results_file = output_dir / "results.json"
    existing_results = []
    try:
        with open(results_file, "r") as f:
            existing_results = json.load(f)
        existing_names = {r["name"] for r in existing_results}
        print(f"Found {len(existing_results)} existing results")
    except FileNotFoundError:
        existing_names = set()

    # Filter to only run configs not yet completed
    configs_to_run = {name: cfg for name, cfg in configs.items() if name not in existing_names}
    print(f"Running {len(configs_to_run)} new configs out of {len(configs)} total")
    print()

    # Print search space summary
    print("Search space (tighter bounds around trial_099):")
    print("  lr: 1e-3 to 2e-3")
    print("  clip: 0.15 to 0.30")
    print("  ent: 0.05 to 0.15")
    print("  vf_coef: 0.25 to 0.50")
    print("  gae_lambda: 0.85 to 0.95")
    print("  max_grad_norm: 0.25 to 0.50")
    print("  n_steps: [32, 64 (weighted), 128]")
    print("  n_epochs: [2 (weighted), 4]")
    print("  ortho_init: True (fixed)")
    print("  lr_anneal: True (fixed)")
    print("  normalize_returns: True (fixed)")
    print()

    # Run each config
    new_results = []
    for name, config in configs_to_run.items():
        print(f"\n{'=' * 60}")
        print(f"Running config: {name}")
        print(f"{'=' * 60}")
        print(f"  lr={config.lr:.2e}, clip={config.clip_eps:.3f}, ent={config.ent_coef:.3f}")
        print(
            f"  n_steps={config.n_steps}, n_epochs={config.n_epochs}, vf_coef={config.vf_coef:.3f}"
        )

        runner_state, metrics = train(
            config,
            ENV_CONFIG,
            seed=42,
            verbose=True,
            log_dir=str(output_dir / name),
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

        print(f"\n  {name} final prey_alive: {final_prey:.3f}, KL: {final_kl:.4f}")

    # Combine with existing results
    all_results = existing_results + new_results

    # Sort by prey_alive (lower is better)
    all_results.sort(key=lambda x: x["prey_alive"])

    # Save configs
    config_summary = {}
    for name, cfg in configs.items():
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
    with open(output_dir / "configs.json", "w") as f:
        json.dump(config_summary, f, indent=2)

    # Save results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print top results
    print("\n" + "=" * 70)
    print("TOP 20 RESULTS (sorted by prey_alive)")
    print("=" * 70)
    print(
        f"{'Rank':<6}{'Name':<12}{'Prey':<8}{'KL':<10}{'LR':<12}{'Clip':<8}{'Ent':<8}{'n_steps':<8}{'n_epochs':<6}"
    )
    print("-" * 80)
    for rank, r in enumerate(all_results[:20], 1):
        print(
            f"{rank:<6}{r['name']:<12}{r['prey_alive']:<8.3f}"
            f"{r['kl']:<10.4f}{r['lr']:<12.2e}{r['clip']:<8.3f}"
            f"{r['ent']:<8.3f}{r['n_steps']:<8}{r['n_epochs']:<6}"
        )

    # Compare with previous best
    print("\n" + "=" * 70)
    print("Comparison with previous best (trial_099 from expanded sweep)")
    print("=" * 70)
    print("Previous best: prey_alive=1.415 (lr=1.3e-3, clip=0.219, ent=0.091)")
    if all_results:
        best = all_results[0]
        improvement = 1.415 - best["prey_alive"]
        print(f"New best: {best['name']} prey_alive={best['prey_alive']:.3f}")
        print(
            f"  Improvement: {improvement:+.3f} (lr={best['lr']:.2e}, clip={best['clip']:.3f}, ent={best['ent']:.3f})"
        )

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
