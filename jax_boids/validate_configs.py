"""Validate top configs from sweep with extended training."""

import json
import argparse
from pathlib import Path

from jax_boids.train_single import TrainConfig, train
from jax_boids.envs.types import EnvConfig


def load_top_configs(results_path: str, top_n: int = 10) -> list[dict]:
    """Load top N configs from sweep results."""
    with open(results_path) as f:
        configs = json.load(f)
    # Already sorted by prey_alive in results.json
    return configs[:top_n]


def run_validation(config: dict, output_dir: Path, seed: int = 123):
    """Run extended training on a single config."""
    run_name = config["name"]
    run_path = output_dir / run_name

    # Convert sweep config to TrainConfig
    train_config = TrainConfig(
        lr=config["lr"],
        clip_eps=config["clip"],
        ent_coef=config["ent"],
        vf_coef=config["vf_coef"],
        gae_lambda=config["gae_lambda"],
        max_grad_norm=config["max_grad_norm"],
        n_steps=config["n_steps"],
        n_epochs=config["n_epochs"],
        total_timesteps=2_000_000,  # Extended: 2M steps vs original sweep
        n_envs=32,
        orthogonal_init=config["orthogonal_init"],
        lr_anneal=config["lr_anneal"],
        min_lr=config["min_lr"],
        normalize_returns=config["normalize_returns"],
    )

    # Simple env for validation
    env_config = EnvConfig(
        n_predators=1,
        n_prey=3,
        world_size=10,
        max_steps=100,
    )

    print(f"\n{'=' * 60}")
    print(f"Validating {run_name} (original prey_alive: {config['prey_alive']:.3f})")
    print(f"Output: {run_path}")
    print(f"{'=' * 60}\n")

    train(train_config, env_config, seed=seed, verbose=True, log_dir=str(run_path))


def main():
    parser = argparse.ArgumentParser(description="Validate top configs from sweep")
    parser.add_argument(
        "--results",
        type=str,
        default="runs/expanded_random_sweep/results.json",
        help="Path to sweep results JSON",
    )
    parser.add_argument("--top", type=int, default=10, help="Number of top configs to validate")
    parser.add_argument(
        "--output", type=str, default="runs/validation", help="Output directory for validation runs"
    )
    args = parser.parse_args()

    configs = load_top_configs(args.results, args.top)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save which configs we're validating
    with open(output_dir / "configs.json", "w") as f:
        json.dump(configs, f, indent=2)

    print(f"Validating top {len(configs)} configs:")
    for c in configs:
        print(f"  {c['name']}: prey_alive={c['prey_alive']:.3f}")

    for config in configs:
        try:
            run_validation(config, output_dir)
        except Exception as e:
            print(f"Failed {config['name']}: {e}")
            continue

    print("\nValidation complete!")


if __name__ == "__main__":
    main()
