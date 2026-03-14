"""Unified training entrypoint for PPO hyperparameter search and validation.

Modes:
- train: Single training run with named config or inline hyperparams
- sweep: Random hyperparameter sweep
- sweep-fine: Fine-tuning sweep around base config
- validate: Extended validation with multi-seed support
"""

import argparse
import json
import random
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from jax_boids.configs import CONFIGS, Config, get_config
from jax_boids.envs.types import EnvConfig, TrainConfig
from jax_boids.train_single import train


class Mode(Enum):
    """Training mode."""

    TRAIN = "train"
    SWEEP = "sweep"
    SWEEP_FINE = "sweep-fine"
    VALIDATE = "validate"


# Default sweep ranges for random sampling
DEFAULT_SWEEP_RANGES = {
    "lr": {"type": "log", "min": 1e-5, "max": 1e-1},
    "clip_eps": {"type": "uniform", "min": 0.05, "max": 0.5},
    "ent_coef": {"type": "uniform", "min": 0.0, "max": 0.2},
    "vf_coef": {"type": "uniform", "min": 0.25, "max": 1.0},
    "gae_lambda": {"type": "uniform", "min": 0.85, "max": 0.995},
    "max_grad_norm": {"type": "uniform", "min": 0.1, "max": 1.0},
    "n_steps": {"type": "choice", "values": [64, 128, 256, 512]},
    "n_epochs": {"type": "choice", "values": [2, 4, 8, 10, 16]},
    "orthogonal_init": {"type": "choice", "values": [True, False]},
    "lr_anneal": {"type": "choice", "values": [True, False]},
    "normalize_returns": {"type": "choice", "values": [True, False]},
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Unified training entrypoint for PPO hyperparameter search and validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  train       Single training run with named config or inline hyperparams
  sweep       Random hyperparameter sweep
  sweep-fine  Fine-tuning sweep around base config
  validate    Extended validation with multi-seed support

Examples:
  # Train a named config (1M steps default)
  python -m jax_boids.train --mode train --config validated_005

  # Train with custom hyperparams
  python -m jax_boids.train --mode train --lr 1e-4 --n-steps 128 --total-timesteps 500000

  # Random sweep (100 configs, 200k steps each)
  python -m jax_boids.train --mode sweep --n-configs 100 --total-timesteps 200000

  # Fine-tune sweep around validated_005
  python -m jax_boids.train --mode sweep-fine --base-config validated_005 --n-configs 50

  # Validate with 5 seeds
  python -m jax_boids.train --mode validate --config validated_005 --n-seeds 5

  # Validate top 10 from sweep results
  python -m jax_boids.train --mode validate --from-results runs/sweep/results.json --top 10
        """,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "sweep", "sweep-fine", "validate"],
        help="Training mode",
    )

    # Config selection
    parser.add_argument("--config", type=str, nargs="+", help="Named config(s) from configs.py")
    parser.add_argument(
        "--from-results",
        type=str,
        help="Load top N configs from sweep results JSON",
    )
    parser.add_argument("--top", type=int, default=10, help="Number of top configs to load")
    parser.add_argument("--base-config", type=str, help="Base config for sweep-fine mode")

    # Training parameters (override config defaults)
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--clip", type=float, help="PPO clip epsilon")
    parser.add_argument("--ent", type=float, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, help="Value function coefficient")
    parser.add_argument("--gae-lambda", type=float, help="GAE lambda")
    parser.add_argument("--max-grad-norm", type=float, help="Max gradient norm")
    parser.add_argument("--n-steps", type=int, help="Steps per rollout")
    parser.add_argument("--n-epochs", type=int, help="PPO epochs")
    parser.add_argument("--n-minibatches", type=int, default=4, help="PPO minibatches")

    # Scale
    parser.add_argument("--total-timesteps", type=int, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, help="Number of parallel environments")

    # Features
    parser.add_argument(
        "--orthogonal-init", action="store_true", help="Use orthogonal initialization"
    )
    parser.add_argument(
        "--no-orthogonal-init", action="store_true", help="Disable orthogonal initialization"
    )
    parser.add_argument("--lr-anneal", action="store_true", help="Enable learning rate annealing")
    parser.add_argument(
        "--no-lr-anneal", action="store_true", help="Disable learning rate annealing"
    )
    parser.add_argument("--min-lr", type=float, default=0.0, help="Minimum LR for annealing")
    parser.add_argument("--normalize-returns", action="store_true", help="Normalize returns")
    parser.add_argument(
        "--no-normalize-returns", action="store_true", help="Disable return normalization"
    )

    # Environment
    parser.add_argument("--n-predators", type=int, default=1, help="Number of predators")
    parser.add_argument("--n-prey", type=int, default=3, help="Number of prey")
    parser.add_argument("--world-size", type=float, default=10.0, help="World size")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")

    # Sweep-specific
    parser.add_argument("--n-configs", type=int, default=100, help="Number of configs for sweep")
    parser.add_argument(
        "--sweep-range",
        type=str,
        help="JSON file with custom sweep ranges",
    )
    parser.add_argument(
        "--perturb-factor",
        type=float,
        default=0.2,
        help="Perturbation factor for sweep-fine (e.g., 0.2 = ±20%%)",
    )

    # Validation-specific
    parser.add_argument("--n-seeds", type=int, default=1, help="Number of seeds per config")
    parser.add_argument("--seeds", type=int, nargs="+", help="Explicit seed list")

    # General
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    parser.add_argument("--log-interval", type=int, default=50, help="Logging frequency")

    return parser.parse_args()


def sample_param(param_range: Dict[str, Any]) -> Any:
    """Sample a parameter value from a range specification."""
    param_type = param_range["type"]

    if param_type == "log":
        import math

        log_min = math.log10(param_range["min"])
        log_max = math.log10(param_range["max"])
        return 10 ** random.uniform(log_min, log_max)
    elif param_type == "uniform":
        return random.uniform(param_range["min"], param_range["max"])
    elif param_type == "choice":
        return random.choice(param_range["values"])
    else:
        raise ValueError(f"Unknown param type: {param_type}")


def generate_sweep_configs(
    n_configs: int,
    sweep_ranges: Optional[Dict[str, Any]] = None,
    total_timesteps: int = 200_000,
    n_envs: int = 64,
) -> Dict[str, TrainConfig]:
    """Generate random configs for sweep."""
    if sweep_ranges is None:
        sweep_ranges = DEFAULT_SWEEP_RANGES.copy()

    configs = {}
    for i in range(n_configs):
        lr = sample_param(sweep_ranges["lr"])

        # Sample min_lr_factor only if lr_anneal is True
        lr_anneal = sample_param(sweep_ranges["lr_anneal"])
        if lr_anneal:
            min_lr_factor = random.choice([0.0, 0.1, 0.3])
            min_lr = min_lr_factor * lr
        else:
            min_lr = 0.0

        configs[f"trial_{i:03d}"] = TrainConfig(
            lr=lr,
            clip_eps=sample_param(sweep_ranges["clip_eps"]),
            ent_coef=sample_param(sweep_ranges["ent_coef"]),
            vf_coef=sample_param(sweep_ranges["vf_coef"]),
            gae_lambda=sample_param(sweep_ranges["gae_lambda"]),
            max_grad_norm=sample_param(sweep_ranges["max_grad_norm"]),
            n_steps=sample_param(sweep_ranges["n_steps"]),
            n_epochs=sample_param(sweep_ranges["n_epochs"]),
            n_minibatches=4,
            n_envs=n_envs,
            total_timesteps=total_timesteps,
            prey_noise_scale=0.1,
            orthogonal_init=sample_param(sweep_ranges["orthogonal_init"]),
            lr_anneal=lr_anneal,
            min_lr=min_lr,
            normalize_returns=sample_param(sweep_ranges["normalize_returns"]),
        )

    return configs


def perturb_value(base_value: Any, perturb_factor: float, param_name: str) -> Any:
    """Perturb a value by a factor for fine-tuning sweep."""
    if param_name in ["n_steps", "n_epochs"]:
        # For discrete params, sample from nearby values
        if param_name == "n_steps":
            options = [64, 128, 256, 512]
            current_idx = options.index(base_value) if base_value in options else 1
            nearby = [
                options[max(0, current_idx - 1)],
                base_value,
                options[min(len(options) - 1, current_idx + 1)],
            ]
            return random.choice(nearby)
        elif param_name == "n_epochs":
            options = [2, 4, 8, 10, 16]
            current_idx = options.index(base_value) if base_value in options else 2
            nearby = [
                options[max(0, current_idx - 1)],
                base_value,
                options[min(len(options) - 1, current_idx + 1)],
            ]
            return random.choice(nearby)

    elif param_name in ["orthogonal_init", "lr_anneal", "normalize_returns"]:
        # For boolean params, flip with perturb_factor probability
        if random.random() < perturb_factor:
            return not base_value
        return base_value

    else:
        # For continuous params, sample from [base * (1 - factor), base * (1 + factor)]
        lower = base_value * (1 - perturb_factor)
        upper = base_value * (1 + perturb_factor)
        # Use log-uniform for better coverage
        import math

        if lower > 0:
            log_lower = math.log10(lower)
            log_upper = math.log10(upper)
            return 10 ** random.uniform(log_lower, log_upper)
        return random.uniform(lower, upper)


def generate_fine_tuning_configs(
    base_config: Config,
    n_configs: int,
    perturb_factor: float = 0.2,
    total_timesteps: int = 200_000,
    n_envs: int = 64,
) -> Dict[str, TrainConfig]:
    """Generate configs by perturbing base config."""
    configs = {}

    for i in range(n_configs):
        lr = perturb_value(base_config.lr, perturb_factor, "lr")
        lr_anneal = perturb_value(base_config.lr_anneal, perturb_factor, "lr_anneal")

        if lr_anneal:
            min_lr_factor = random.choice([0.0, 0.1, 0.3])
            min_lr = min_lr_factor * lr
        else:
            min_lr = 0.0

        configs[f"fine_{i:03d}"] = TrainConfig(
            lr=lr,
            clip_eps=perturb_value(base_config.clip_eps, perturb_factor, "clip_eps"),
            ent_coef=perturb_value(base_config.ent_coef, perturb_factor, "ent_coef"),
            vf_coef=perturb_value(base_config.vf_coef, perturb_factor, "vf_coef"),
            gae_lambda=perturb_value(base_config.gae_lambda, perturb_factor, "gae_lambda"),
            max_grad_norm=perturb_value(base_config.max_grad_norm, perturb_factor, "max_grad_norm"),
            n_steps=perturb_value(base_config.n_steps, perturb_factor, "n_steps"),
            n_epochs=perturb_value(base_config.n_epochs, perturb_factor, "n_epochs"),
            n_minibatches=4,
            n_envs=n_envs,
            total_timesteps=total_timesteps,
            prey_noise_scale=0.1,
            orthogonal_init=perturb_value(
                base_config.orthogonal_init, perturb_factor, "orthogonal_init"
            ),
            lr_anneal=lr_anneal,
            min_lr=min_lr,
            normalize_returns=perturb_value(
                base_config.normalize_returns, perturb_factor, "normalize_returns"
            ),
        )

    return configs


def config_to_train_config(
    config: Optional[Config],
    args: argparse.Namespace,
    total_timesteps: Optional[int] = None,
) -> TrainConfig:
    """Convert Config dataclass to TrainConfig, applying CLI overrides."""

    if config is None:
        raise ValueError("config cannot be None")

    lr_anneal = (
        args.lr_anneal
        if args.lr_anneal
        else (not args.no_lr_anneal and getattr(config, "lr_anneal", False))
    )
    orthogonal_init = (
        args.orthogonal_init
        if args.orthogonal_init
        else (not args.no_orthogonal_init and getattr(config, "orthogonal_init", False))
    )
    normalize_returns = (
        args.normalize_returns
        if args.normalize_returns
        else (not args.no_normalize_returns and getattr(config, "normalize_returns", False))
    )
    orthogonal_init = (
        args.orthogonal_init
        if args.orthogonal_init
        else (not args.no_orthogonal_init and getattr(config, "orthogonal_init", False))
    )
    normalize_returns = (
        args.normalize_returns
        if args.normalize_returns
        else (not args.no_normalize_returns and getattr(config, "normalize_returns", False))
    )

    return TrainConfig(
        lr=args.lr if args.lr is not None else config.lr,
        clip_eps=args.clip if args.clip is not None else config.clip_eps,
        ent_coef=args.ent if args.ent is not None else config.ent_coef,
        vf_coef=args.vf_coef if args.vf_coef is not None else config.vf_coef,
        gae_lambda=args.gae_lambda if args.gae_lambda is not None else config.gae_lambda,
        max_grad_norm=args.max_grad_norm
        if args.max_grad_norm is not None
        else config.max_grad_norm,
        n_steps=args.n_steps if args.n_steps is not None else config.n_steps,
        n_epochs=args.n_epochs if args.n_epochs is not None else config.n_epochs,
        n_minibatches=args.n_minibatches,
        n_envs=args.n_envs if args.n_envs is not None else 32,
        total_timesteps=total_timesteps or args.total_timesteps or 1_000_000,
        prey_noise_scale=0.1,
        orthogonal_init=orthogonal_init,
        lr_anneal=lr_anneal,
        min_lr=args.min_lr if args.min_lr else getattr(config, "min_lr", 0.0),
        normalize_returns=normalize_returns,
    )


def get_env_config(args: argparse.Namespace) -> EnvConfig:
    """Get environment config from CLI args."""
    return EnvConfig(
        n_predators=args.n_predators,
        n_prey=args.n_prey,
        world_size=args.world_size,
        max_steps=args.max_steps,
    )


def run_training(
    config: TrainConfig,
    env_config: EnvConfig,
    seed: int,
    output_dir: Path,
    verbose: bool = True,
    log_interval: int = 50,
    log_dir: Optional[str] = None,
) -> Dict:
    """Run single training job. Returns metrics dict.

    Args:
        log_dir: Directory for TensorBoard logs and checkpoints.
            None disables logging (used for sweeps to save memory).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = {
        "lr": config.lr,
        "clip_eps": config.clip_eps,
        "ent_coef": config.ent_coef,
        "vf_coef": config.vf_coef,
        "gae_lambda": config.gae_lambda,
        "max_grad_norm": config.max_grad_norm,
        "n_steps": config.n_steps,
        "n_epochs": config.n_epochs,
        "n_minibatches": config.n_minibatches,
        "n_envs": config.n_envs,
        "total_timesteps": config.total_timesteps,
        "orthogonal_init": config.orthogonal_init,
        "lr_anneal": config.lr_anneal,
        "min_lr": config.min_lr,
        "normalize_returns": config.normalize_returns,
        "seed": seed,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    runner_state, metrics = train(
        config,
        env_config,
        seed=seed,
        verbose=verbose,
        log_dir=log_dir,
    )

    # Extract final metrics
    final_metrics = {
        "prey_alive": float(metrics["prey_alive"][-1]),
        "reward": float(metrics["reward"][-1]),
        "policy_loss": float(metrics["policy_loss"][-1]),
        "value_loss": float(metrics["value_loss"][-1]),
        "entropy": float(metrics["entropy"][-1]),
        "approx_kl": float(metrics["approx_kl"][-1]),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    return final_metrics


def mode_train(args: argparse.Namespace) -> None:
    """Single training run."""
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    elif args.config and len(args.config) == 1:
        output_dir = Path(f"runs/train_{args.config[0]}")
    else:
        output_dir = Path("runs/train")

    config = None
    if args.config:
        if len(args.config) > 1:
            print("Error: --mode train accepts only one config. Use --mode validate for multiple.")
            return
        config = get_config(args.config[0])
        print(f"Training with config: {args.config[0]}")
        print(f"  lr={config.lr:.2e} clip={config.clip_eps:.3f} ent={config.ent_coef:.3f}")
        print(f"  n_steps={config.n_steps} n_epochs={config.n_epochs}")
        train_config = config_to_train_config(config, args)
    else:
        # Build config entirely from CLI args (inline hyperparams)
        print("Training with inline hyperparams")

        # Resolve boolean flags
        orthogonal_init = args.orthogonal_init and not args.no_orthogonal_init
        lr_anneal = args.lr_anneal and not args.no_lr_anneal
        normalize_returns = args.normalize_returns and not args.no_normalize_returns

        train_config = TrainConfig(
            lr=args.lr or 3e-4,
            clip_eps=args.clip or 0.2,
            ent_coef=args.ent or 0.01,
            vf_coef=args.vf_coef or 0.5,
            gae_lambda=args.gae_lambda or 0.95,
            max_grad_norm=args.max_grad_norm or 0.5,
            n_steps=args.n_steps or 128,
            n_epochs=args.n_epochs or 4,
            n_minibatches=args.n_minibatches,
            n_envs=args.n_envs or 32,
            total_timesteps=args.total_timesteps or 1_000_000,
            prey_noise_scale=0.1,
            orthogonal_init=orthogonal_init,
            lr_anneal=lr_anneal,
            min_lr=args.min_lr,
            normalize_returns=normalize_returns,
        )
        print(
            f"  lr={train_config.lr:.2e} clip={train_config.clip_eps:.3f} ent={train_config.ent_coef:.3f}"
        )
        print(f"  n_steps={train_config.n_steps} n_epochs={train_config.n_epochs}")
    env_config = get_env_config(args)

    run_training(
        train_config,
        env_config,
        seed=args.seed,
        output_dir=output_dir,
        verbose=args.verbose,
        log_interval=args.log_interval,
        log_dir=str(output_dir),
    )

    print(f"\nTraining complete! Output: {output_dir}")


def mode_sweep(args: argparse.Namespace) -> None:
    """Random hyperparameter sweep."""
    output_dir = Path(args.output or "runs/sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load custom sweep ranges if provided
    sweep_ranges = None
    if args.sweep_range:
        with open(args.sweep_range) as f:
            sweep_ranges = json.load(f)
        print(f"Using custom sweep ranges from {args.sweep_range}")

    # Generate configs
    random.seed(args.seed)
    total_timesteps = args.total_timesteps or 200_000
    configs = generate_sweep_configs(args.n_configs, sweep_ranges, total_timesteps=total_timesteps)

    # Load existing results
    results_path = output_dir / "results.json"
    existing_results = []
    existing_names = set()
    if results_path.exists():
        with open(results_path) as f:
            existing_results = json.load(f)
        existing_names = {r["name"] for r in existing_results}
        print(f"Found {len(existing_results)} existing results")

    # Filter to only run configs not yet completed
    configs_to_run = {name: cfg for name, cfg in configs.items() if name not in existing_names}
    print(f"Running {len(configs_to_run)} new configs out of {len(configs)} total\n")

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

        run_dir = output_dir / name
        env_config = get_env_config(args)

        final_metrics = run_training(
            config,
            env_config,
            seed=args.seed,
            output_dir=run_dir,
            verbose=args.verbose,
            log_interval=args.log_interval,
        )

        result = {
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
            "prey_alive": final_metrics["prey_alive"],
            "kl": final_metrics["approx_kl"],
        }
        new_results.append(result)

        print(
            f"\n  {name} final prey_alive: {final_metrics['prey_alive']:.2f}, "
            f"KL: {final_metrics['approx_kl']:.4f}"
        )

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
    with open(results_path, "w") as f:
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

    print(f"\nResults saved to {results_path}")


def mode_sweep_fine(args: argparse.Namespace) -> None:
    """Fine-tuning sweep around base config."""
    if not args.base_config:
        print("Error: --mode sweep-fine requires --base-config")
        return

    if args.base_config not in CONFIGS:
        print(f"Error: Unknown config '{args.base_config}'. Available: {list(CONFIGS.keys())}")
        return

    base_config = get_config(args.base_config)
    print(f"Fine-tuning around config: {args.base_config}")
    print(
        f"  lr={base_config.lr:.2e} clip={base_config.clip_eps:.3f} ent={base_config.ent_coef:.3f}"
    )
    print(f"  n_steps={base_config.n_steps} n_epochs={base_config.n_epochs}")
    print(f"  Perturb factor: {args.perturb_factor * 100:.0f}%\n")

    # Generate fine-tuning configs
    random.seed(args.seed)
    total_timesteps = args.total_timesteps or 200_000
    configs = generate_fine_tuning_configs(
        base_config, args.n_configs, args.perturb_factor, total_timesteps=total_timesteps
    )

    # Set output directory
    output_dir = Path(args.output or f"runs/fine_tuning_{args.base_config}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save base config info
    with open(output_dir / "base_config.json", "w") as f:
        json.dump(
            {
                "name": args.base_config,
                "lr": base_config.lr,
                "clip_eps": base_config.clip_eps,
                "ent_coef": base_config.ent_coef,
                "vf_coef": base_config.vf_coef,
                "gae_lambda": base_config.gae_lambda,
                "max_grad_norm": base_config.max_grad_norm,
                "n_steps": base_config.n_steps,
                "n_epochs": base_config.n_epochs,
                "orthogonal_init": base_config.orthogonal_init,
                "lr_anneal": base_config.lr_anneal,
                "min_lr": base_config.min_lr,
                "normalize_returns": base_config.normalize_returns,
                "perturb_factor": args.perturb_factor,
            },
            f,
            indent=2,
        )

    # Load existing results
    results_path = output_dir / "results.json"
    existing_results = []
    existing_names = set()
    if results_path.exists():
        with open(results_path) as f:
            existing_results = json.load(f)
        existing_names = {r["name"] for r in existing_results}
        print(f"Found {len(existing_results)} existing results")

    # Filter to only run configs not yet completed
    configs_to_run = {name: cfg for name, cfg in configs.items() if name not in existing_names}
    print(f"Running {len(configs_to_run)} new configs out of {len(configs)} total\n")

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

        run_dir = output_dir / name
        env_config = get_env_config(args)

        final_metrics = run_training(
            config,
            env_config,
            seed=args.seed,
            output_dir=run_dir,
            verbose=args.verbose,
        )

        result = {
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
            "prey_alive": final_metrics["prey_alive"],
            "kl": final_metrics["approx_kl"],
        }
        new_results.append(result)

        print(
            f"\n  {name} final prey_alive: {final_metrics['prey_alive']:.2f}, "
            f"KL: {final_metrics['approx_kl']:.4f}"
        )

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
    with open(results_path, "w") as f:
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

    print(f"\nResults saved to {results_path}")


def mode_validate(args: argparse.Namespace) -> None:
    """Extended validation with multi-seed support."""
    output_dir = Path(args.output or "runs/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which configs to validate
    configs_to_validate = []

    if args.config:
        for config_name in args.config:
            if config_name in CONFIGS:
                configs_to_validate.append(("named", config_name, None))
                cfg = CONFIGS[config_name]
                print(f"Validating named config: {config_name}")
                print(f"  lr={cfg.lr:.2e} clip={cfg.clip_eps:.3f} ent={cfg.ent_coef:.3f}")
            else:
                print(f"Warning: Unknown config '{config_name}'. Skipping.")

    if args.from_results:
        with open(args.from_results) as f:
            results = json.load(f)
        top_configs = results[: args.top]
        for cfg in top_configs:
            configs_to_validate.append(("sweep", cfg["name"], cfg))
        print(f"\nValidating top {len(top_configs)} configs from {args.from_results}")

    if not configs_to_validate:
        print("Error: Must specify --config or --from-results")
        return

    # Generate seeds
    if args.seeds:
        seeds = args.seeds
    else:
        seeds = list(range(123, 123 + args.n_seeds))

    print(f"\nSeeds: {seeds}")
    total_runs = len(configs_to_validate) * len(seeds)
    print(f"Total runs: {len(configs_to_validate)} configs × {len(seeds)} seeds = {total_runs}")

    # Save which configs we're validating
    validation_config = []
    for source, name, cfg in configs_to_validate:
        validation_config.append({"name": name, "source": source})
    with open(output_dir / "configs.json", "w") as f:
        json.dump(validation_config, f, indent=2)

    # Run validation
    all_results = []
    default_timesteps = 2_000_000  # Extended training for validation

    for source, config_name, sweep_cfg in configs_to_validate:
        if source == "named":
            config = get_config(config_name)
        else:
            # Create a temporary Config object from sweep results
            from jax_boids.configs import Config as ConfigDataclass

            config = ConfigDataclass(
                name=sweep_cfg["name"],
                source=f"sweep_{sweep_cfg['name']}",
                lr=sweep_cfg["lr"],
                clip_eps=sweep_cfg["clip"],
                ent_coef=sweep_cfg["ent"],
                vf_coef=sweep_cfg["vf_coef"],
                gae_lambda=sweep_cfg["gae_lambda"],
                max_grad_norm=sweep_cfg["max_grad_norm"],
                n_steps=sweep_cfg["n_steps"],
                n_epochs=sweep_cfg["n_epochs"],
                orthogonal_init=sweep_cfg["orthogonal_init"],
                lr_anneal=sweep_cfg["lr_anneal"],
                min_lr=sweep_cfg.get("min_lr", 0.0),
                normalize_returns=sweep_cfg["normalize_returns"],
            )

        for seed in seeds:
            run_name = f"{config_name}_seed{seed}"
            run_dir = output_dir / run_name

            print(f"\n{'=' * 60}")
            print(f"Validating {run_name}")
            print(f"{'=' * 60}")

            train_config = config_to_train_config(config, args, total_timesteps=default_timesteps)
            env_config = get_env_config(args)

            final_metrics = run_training(
                train_config,
                env_config,
                seed=seed,
                output_dir=run_dir,
                verbose=args.verbose,
                log_interval=args.log_interval,
            )

            result = {
                "name": config_name,
                "seed": seed,
                "prey_alive": final_metrics["prey_alive"],
                "lr": train_config.lr,
                "clip": train_config.clip_eps,
                "ent": train_config.ent_coef,
                "n_steps": train_config.n_steps,
                "n_epochs": train_config.n_epochs,
            }
            all_results.append(result)

            print(f"  {run_name} final prey_alive: {final_metrics['prey_alive']:.2f}")

    # Aggregate results by config
    aggregated = {}
    for r in all_results:
        name = r["name"]
        if name not in aggregated:
            aggregated[name] = {"name": name, "seeds": [], "prey_alive": []}
        aggregated[name]["seeds"].append(r["seed"])
        aggregated[name]["prey_alive"].append(r["prey_alive"])

    for name, data in aggregated.items():
        data["mean"] = sum(data["prey_alive"]) / len(data["prey_alive"])
        data["std"] = (
            sum((x - data["mean"]) ** 2 for x in data["prey_alive"]) / len(data["prey_alive"])
        ) ** 0.5

    aggregated_list = list(aggregated.values())
    aggregated_list.sort(key=lambda x: x["mean"])

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(aggregated_list, f, indent=2)

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS (sorted by mean prey_alive)")
    print("=" * 60)
    print(f"{'Config':<20}{'Mean':<12}{'Std':<12}{'Seeds':<10}")
    print("-" * 60)
    for r in aggregated_list:
        print(f"{r['name']:<20}{r['mean']:<12.3f}{r['std']:<12.3f}{len(r['seeds']):<10}")

    print(f"\nResults saved to {output_dir / 'results.json'}")


def main():
    args = parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Dispatch to appropriate mode
    if args.mode == "train":
        mode_train(args)
    elif args.mode == "sweep":
        mode_sweep(args)
    elif args.mode == "sweep-fine":
        mode_sweep_fine(args)
    elif args.mode == "validate":
        mode_validate(args)


if __name__ == "__main__":
    main()
