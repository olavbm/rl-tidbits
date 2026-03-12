#!/usr/bin/env python3
"""Analyze random sweep results and correlate with training diagnostics."""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jax_boids.telemetry.diagnostics import analyze_run


def load_sweep_results(sweep_path: str) -> tuple[list[dict[str, Any]], str]:
    """Load results.json from sweep directory or file."""
    path = Path(sweep_path)

    # If it's a file, use it directly
    if path.is_file():
        results_path = path
        sweep_dir = str(path.parent)
    else:
        # Assume it's a directory, look for results.json
        results_path = path / "results.json"
        sweep_dir = str(path)

    if not results_path.exists():
        print(f"Error: {results_path} not found")
        sys.exit(1)

    with open(results_path) as f:
        return json.load(f), sweep_dir


def extract_config_name(result: dict[str, Any]) -> str:
    """Create a readable config identifier."""
    parts = []
    if result.get("orthogonal_init"):
        parts.append("ortho")
    if result.get("lr_anneal"):
        parts.append("anneal")
    if result.get("normalize_returns"):
        parts.append("norm_ret")
    return "_".join(parts) if parts else "base"


def get_config_value(result: dict[str, Any], key: str, default: Any = None) -> Any:
    """Get config value, handling nested or flat structure and alternative names."""
    if "config" in result:
        val = result["config"].get(key)
        if val is not None:
            return val
    val = result.get(key)
    if val is not None:
        return val

    # Handle alternative key names
    alt_keys = {
        "clip_epsilon": "clip",
        "ent_coef": "ent",
        "ortho_init": "orthogonal_init",
    }
    alt_key = alt_keys.get(key)
    if alt_key:
        if "config" in result:
            val = result["config"].get(alt_key)
            if val is not None:
                return val
        val = result.get(alt_key)
        if val is not None:
            return val

    return default


def analyze_sweep(sweep_path: str, top_n: int = 20) -> None:
    """Analyze sweep results with diagnostics."""
    results, sweep_dir = load_sweep_results(sweep_path)

    # Sort by prey_alive (lower is better)
    sorted_results = sorted(results, key=lambda x: x["prey_alive"])

    print(f"\n{'=' * 80}")
    print(f"Sweep Analysis: {sweep_dir}")
    print(f"Total trials: {len(results)}")
    print(f"{'=' * 80}\n")

    # Summary statistics
    prey_values = [r["prey_alive"] for r in results]
    print("Summary Statistics:")
    print(f"  Best prey_alive: {min(prey_values):.3f}")
    print(f"  Worst prey_alive: {max(prey_values):.3f}")
    print(f"  Mean prey_alive: {np.mean(prey_values):.3f}")
    print(f"  Std prey_alive: {np.std(prey_values):.3f}")
    print(f"  Below target (<1.5): {sum(1 for p in prey_values if p < 1.5)}")
    print()

    # Top configs table
    print(f"Top {top_n} Configs by prey_alive:")
    print("-" * 80)
    print(
        f"{'Rank':<6} {'prey_alive':<12} {'lr':<12} {'clip':<8} {'ent':<8} "
        f"{'n_steps':<8} {'n_epochs':<8} {'config':<15}"
    )
    print("-" * 80)

    for i, result in enumerate(sorted_results[:top_n], 1):
        config_name = extract_config_name(result)
        lr = get_config_value(result, "lr")
        clip = get_config_value(result, "clip_epsilon")
        ent = get_config_value(result, "ent_coef")
        n_steps = get_config_value(result, "n_steps")
        n_epochs = get_config_value(result, "n_epochs")
        print(
            f"{i:<6} {result['prey_alive']:<12.3f} "
            f"{lr:<12.3e} {clip:<8.3f} {ent:<8.3f} {n_steps:<8} "
            f"{n_epochs:<8} {config_name:<15}"
        )

    print()

    # Detailed analysis of top 5
    print("Detailed Analysis of Top 5 Configs:")
    print("=" * 80)

    for i, result in enumerate(sorted_results[:5], 1):
        trial_name = result.get("trial_name", result.get("name", f"trial_{i:03d}"))

        print(f"\n{i}. {trial_name} (prey_alive: {result['prey_alive']:.3f})")
        print("-" * 60)

        # Config details
        print("Hyperparameters:")
        print(f"  Learning rate: {get_config_value(result, 'lr'):.3e}")
        print(f"  Clip epsilon: {get_config_value(result, 'clip_epsilon'):.3f}")
        print(f"  Entropy coef: {get_config_value(result, 'ent_coef'):.3f}")
        print(f"  VF coef: {get_config_value(result, 'vf_coef'):.3f}")
        print(f"  GAE lambda: {get_config_value(result, 'gae_lambda'):.3f}")
        print(f"  Max grad norm: {get_config_value(result, 'max_grad_norm'):.3f}")
        print(f"  N steps: {get_config_value(result, 'n_steps')}")
        print(f"  N epochs: {get_config_value(result, 'n_epochs')}")
        print(f"  Min LR: {get_config_value(result, 'min_lr', 'N/A')}")
        print(f"  Ortho init: {get_config_value(result, 'orthogonal_init', False)}")
        print(f"  LR anneal: {get_config_value(result, 'lr_anneal', False)}")
        print(f"  Normalize returns: {get_config_value(result, 'normalize_returns', False)}")

        # Try to get diagnostics
        log_dir = Path(sweep_dir) / trial_name
        if log_dir.exists():
            try:
                analysis = analyze_run(str(log_dir), "pred")
                print(f"\n  Training health: {analysis.learning_assessment}")

                # Show key metrics
                for tag, stats in analysis.metrics_summary.items():
                    if "reward" in tag or "value_loss" in tag:
                        if "mean" in stats:
                            print(f"  {tag}: mean={stats['mean']:.3f}")
            except Exception as e:
                print(f"  Could not analyze logs: {e}")
        else:
            print(f"  Log directory not found: {log_dir}")

    # Correlation analysis
    print("\n" + "=" * 80)
    print("Correlation Analysis:")
    print("-" * 80)

    df = pd.DataFrame(results)
    # Handle both naming conventions
    if "clip" in df.columns:
        df["clip_epsilon"] = df["clip"]
    if "ent" in df.columns:
        df["ent_coef"] = df["ent"]

    numeric_cols = ["prey_alive", "lr", "clip_epsilon", "ent_coef", "vf_coef", "gae_lambda"]
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    corr_matrix = df[numeric_cols].corr()
    prey_corr = corr_matrix["prey_alive"].drop("prey_alive").sort_values()

    print("\nCorrelations with prey_alive:")
    for param, corr in prey_corr.items():
        sign = "↑" if corr > 0 else "↓"
        print(f"  {param}: {corr:+.3f} {sign}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze random sweep results")
    parser.add_argument("sweep_dir", help="Path to sweep directory")
    parser.add_argument("--top", type=int, default=20, help="Number of top configs to show")

    args = parser.parse_args()
    analyze_sweep(args.sweep_dir, args.top)
