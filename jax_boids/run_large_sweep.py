"""Large-scale random hyperparameter search."""

import json
import os
from pathlib import Path
from typing import Any

import jax
from jax import random

from jax_boids.envs.types import EnvConfig
from jax_boids.train_single import TrainConfig, train


def sample_config(seed: int) -> dict[str, Any]:
    """Sample a random config from expanded search space."""
    key = random.PRNGKey(seed)

    keys = random.split(key, 10)

    return {
        # Learning rate: log-uniform from 1e-4 to 1e-2
        "lr": float(random.uniform(keys[0], minval=1e-4, maxval=1e-2)),
        # Clip: uniform from 0.1 to 0.4
        "clip": float(random.uniform(keys[1], minval=0.1, maxval=0.4)),
        # Entropy coef: uniform from 0.0 to 0.1
        "ent_coef": float(random.uniform(keys[2], minval=0.0, maxval=0.1)),
        # Value loss coef: uniform from 0.25 to 1.0
        "vf_coef": float(random.uniform(keys[3], minval=0.25, maxval=1.0)),
        # GAE lambda: uniform from 0.9 to 0.99
        "gae_lambda": float(random.uniform(keys[4], minval=0.9, maxval=0.99)),
        # Discount: fixed at 0.99
        "gamma": 0.99,
        # Steps per update: sample from discrete options
        "n_steps": [64, 128, 256, 512][int(random.randint(keys[5], shape=(), minval=0, maxval=4))],
        # Epochs: sample from discrete options
        "n_epochs": [2, 4, 8, 10, 16][int(random.randint(keys[6], shape=(), minval=0, maxval=5))],
        # Max grad norm: uniform from 0.1 to 1.0
        "max_grad_norm": float(random.uniform(keys[7], shape=(), minval=0.1, maxval=1.0)),
        # Orthogonal init: 50% chance
        "orthogonal_init": bool(random.randint(keys[8], shape=(), minval=0, maxval=2)),
        # LR anneal: 50% chance
        "lr_anneal": bool(random.randint(keys[9], shape=(), minval=0, maxval=2)),
        # Normalize returns: 50% chance
        "normalize_returns": bool(random.randint(keys[9], shape=(), minval=0, maxval=2)),
    }


def run_single_trial(trial_id: int, config: dict, output_dir: Path) -> dict[str, Any]:
    """Run a single trial."""
    run_dir = output_dir / f"trial_{trial_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Simple environment for fast evaluation
    env_config = EnvConfig(
        n_predators=1,
        n_prey=3,
        world_size=10.0,
        max_steps=100,
        prey_learn=False,
        distance_reward=False,
    )

    train_config = TrainConfig(
        lr=config["lr"],
        clip_eps=config["clip"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        gae_lambda=config["gae_lambda"],
        gamma=config["gamma"],
        n_steps=config["n_steps"],
        n_epochs=config["n_epochs"],
        max_grad_norm=config["max_grad_norm"],
        total_timesteps=int(200_000),  # 200k steps per trial
        n_envs=16,
        orthogonal_init=config["orthogonal_init"],
        lr_anneal=config["lr_anneal"],
        normalize_returns=config["normalize_returns"],
    )

    _, metrics = train(
        train_config,
        env_config,
        seed=trial_id,
        verbose=False,
        log_dir=str(run_dir),
    )

    result = {
        "trial_id": trial_id,
        "config": config,
        "prey_alive": float(metrics["prey_alive"][-1]),
        "kl": float(metrics["approx_kl"][-1]),
        "reward": float(metrics["reward"][-1]),
        "policy_loss": float(metrics["policy_loss"][-1]),
    }

    # Save result
    with open(run_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    output_dir = Path("runs/large_random_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing results
    results_file = output_dir / "results.json"
    existing_results = []
    if results_file.exists():
        with open(results_file) as f:
            existing_results = json.load(f)

    existing_trials = {r["trial_id"] for r in existing_results}
    print(f"Found {len(existing_results)} existing results")

    # Generate 100 trial configs
    all_configs = []
    for i in range(100):
        config = sample_config(seed=i + 42)
        all_configs.append({"trial_id": i, "config": config})

    # Save all configs
    with open(output_dir / "configs.json", "w") as f:
        json.dump(all_configs, f, indent=2)

    # Run trials
    results = existing_results.copy()
    for trial_data in all_configs:
        trial_id = trial_data["trial_id"]

        if trial_id in existing_trials:
            print(f"Skipping trial_{trial_id:03d} (already done)")
            continue

        print(f"\n{'=' * 60}")
        print(
            f"Running trial_{trial_id:03d}: LR={trial_data['config']['lr']:.2e}, "
            f"Clip={trial_data['config']['clip']:.2f}, Ent={trial_data['config']['ent_coef']:.3f}, "
            f"n_steps={trial_data['config']['n_steps']}, n_epochs={trial_data['config']['n_epochs']}"
        )
        print(f"{'=' * 60}")

        try:
            result = run_single_trial(trial_id, trial_data["config"], output_dir)
            results.append(result)

            # Save running results
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

            print(f"  -> prey_alive: {result['prey_alive']:.2f}, KL: {result['kl']:.4f}")

        except Exception as e:
            print(f"  -> ERROR: {e}")
            results.append(
                {
                    "trial_id": trial_id,
                    "config": trial_data["config"],
                    "error": str(e),
                }
            )
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

    # Sort and print final results
    success_results = [r for r in results if "prey_alive" in r]
    success_results.sort(key=lambda x: x["prey_alive"])

    print(f"\n{'=' * 60}")
    print("TOP 20 RESULTS (sorted by prey_alive)")
    print(f"{'=' * 60}")
    print(f"{'Rank':<6} {'Trial':<12} {'Prey':<8} {'KL':<10} {'LR':<12} {'Clip':<8} {'Ent':<8}")
    print("-" * 60)
    for i, r in enumerate(success_results[:20], 1):
        c = r["config"]
        print(
            f"{i:<6} {r['trial_id']:<12} {r['prey_alive']:<8.2f} {r['kl']:<10.4f} "
            f"{c['lr']:<12.2e} {c['clip']:<8.2f} {c['ent_coef']:<8.3f}"
        )

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
