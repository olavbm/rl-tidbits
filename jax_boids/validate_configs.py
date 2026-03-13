"""Validate configs with extended training and multi-seed support."""

import json
import argparse
from pathlib import Path

from jax_boids.train_single import TrainConfig, train
from jax_boids.envs.types import EnvConfig
from jax_boids.configs import get_config, Config


def load_top_configs(results_path: str, top_n: int = 10) -> list[dict]:
    """Load top N configs from sweep results."""
    with open(results_path) as f:
        configs = json.load(f)
    # Already sorted by prey_alive in results.json
    return configs[:top_n]


def config_to_train_config(cfg: Config) -> TrainConfig:
    """Convert Config dataclass to TrainConfig."""
    return TrainConfig(
        lr=cfg.lr,
        clip_eps=cfg.clip_eps,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        gae_lambda=cfg.gae_lambda,
        max_grad_norm=cfg.max_grad_norm,
        n_steps=cfg.n_steps,
        n_epochs=cfg.n_epochs,
        total_timesteps=2_000_000,
        n_envs=32,
        orthogonal_init=cfg.orthogonal_init,
        lr_anneal=cfg.lr_anneal,
        min_lr=cfg.min_lr,
        normalize_returns=cfg.normalize_returns,
    )


def run_validation(config: dict | Config, output_dir: Path, seed: int = 123):
    """Run extended training on a single config."""
    if isinstance(config, Config):
        run_name = f"{config.name}_seed{seed}"
        train_config = config_to_train_config(config)
        notes = f"seed={seed}"
    else:
        run_name = config["name"]
        train_config = TrainConfig(
            lr=config["lr"],
            clip_eps=config["clip"],
            ent_coef=config["ent"],
            vf_coef=config["vf_coef"],
            gae_lambda=config["gae_lambda"],
            max_grad_norm=config["max_grad_norm"],
            n_steps=config["n_steps"],
            n_epochs=config["n_epochs"],
            total_timesteps=2_000_000,
            n_envs=32,
            orthogonal_init=config["orthogonal_init"],
            lr_anneal=config["lr_anneal"],
            min_lr=config["min_lr"],
            normalize_returns=config["normalize_returns"],
        )
        notes = f"original prey_alive: {config.get('prey_alive', 'N/A')}"

    run_path = output_dir / run_name

    env_config = EnvConfig(
        n_predators=1,
        n_prey=3,
        world_size=10,
        max_steps=100,
    )

    print(f"\n{'=' * 60}")
    print(f"Validating {run_name}")
    print(f"Notes: {notes}")
    print(f"Output: {run_path}")
    print(f"{'=' * 60}\n")

    train(train_config, env_config, seed=seed, verbose=True, log_dir=str(run_path))


def main():
    parser = argparse.ArgumentParser(description="Validate configs with extended training")
    parser.add_argument(
        "--results",
        type=str,
        default="runs/expanded_random_sweep/results.json",
        help="Path to sweep results JSON",
    )
    parser.add_argument(
        "--top", type=int, default=10, help="Number of top configs to validate from sweep"
    )
    parser.add_argument(
        "--config", type=str, help="Named config from configs.py (e.g., validated_005)"
    )
    parser.add_argument(
        "--n-seeds", type=int, default=1, help="Number of seeds to run for each config"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[123], help="Specific seeds to use")
    parser.add_argument(
        "--output", type=str, default="runs/validation", help="Output directory for validation runs"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which configs to validate
    if args.config:
        cfg = get_config(args.config)
        configs = [cfg]
        print(f"Validating named config: {args.config}")
        print(f"  lr={cfg.lr:.2e} clip={cfg.clip_eps:.3f} ent={cfg.ent_coef:.3f}")
        print(f"  n_steps={cfg.n_steps} n_epochs={cfg.n_epochs}")
        print(
            f"  ortho={cfg.orthogonal_init} lr_anneal={cfg.lr_anneal} norm_ret={cfg.normalize_returns}"
        )
    else:
        configs = load_top_configs(args.results, args.top)
        print(f"Validating top {len(configs)} configs from sweep:")
        for c in configs:
            print(f"  {c['name']}: prey_alive={c['prey_alive']:.3f}")

    # Save which configs we're validating
    with open(output_dir / "configs.json", "w") as f:
        if args.config:
            json.dump([{"name": args.config, "source": "configs.py"}], f, indent=2)
        else:
            json.dump(configs, f, indent=2)

    # Generate seeds if not specified
    seeds = args.seeds if len(args.seeds) == args.n_seeds else list(range(123, 123 + args.n_seeds))

    for config in configs:
        for seed in seeds:
            try:
                run_validation(config, output_dir, seed=seed)
            except Exception as e:
                name = config.name if isinstance(config, Config) else config["name"]
                print(f"Failed {name} seed={seed}: {e}")
                continue

    print("\nValidation complete!")


if __name__ == "__main__":
    main()
