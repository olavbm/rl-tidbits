#!/usr/bin/env python3
"""Convenient script to run diagnostics on curriculum training runs."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_boids.telemetry.diagnostics import compare_runs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run diagnostics on curriculum training runs")
    parser.add_argument(
        "log_dir",
        nargs="?",
        default="runs/curriculum_v2",
        help="Path to curriculum log directory (default: runs/curriculum_v2)",
    )
    parser.add_argument(
        "--agent", default="pred", choices=["pred", "prey"], help="Agent to analyze"
    )
    parser.add_argument(
        "--stage",
        help="Analyze specific stage only (e.g., stage1_easy, stage2_medium)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.stage:
        # Single stage analysis
        stage_dir = os.path.join(args.log_dir, args.stage)
        if not os.path.exists(stage_dir):
            print(f"Stage directory not found: {stage_dir}")
            sys.exit(1)
        compare_runs(stage_dir, args.agent)
    else:
        # All stages analysis
        base_path = args.log_dir
        if not os.path.exists(base_path):
            print(f"Log directory not found: {base_path}")
            sys.exit(1)

        # Find all stage directories
        stages = []
        for entry in sorted(os.listdir(base_path)):
            stage_path = os.path.join(base_path, entry)
            if os.path.isdir(stage_path):
                stages.append(entry)

        if not stages:
            print(f"No stage directories found in {base_path}")
            sys.exit(1)

        print(f"\n{'=' * 80}")
        print(f"Curriculum Diagnostics: {base_path}")
        print(f"Stages: {', '.join(stages)}")
        print(f"{'=' * 80}\n")

        # Track summary stats
        stage_summaries = []

        for stage in stages:
            stage_dir = os.path.join(base_path, stage)
            compare_runs(stage_dir, args.agent)
            print()

            # Get summary for this stage
            from jax_boids.telemetry.reader import read_all_runs
            from jax_boids.telemetry.diagnostics import analyze_run

            runs = read_all_runs(stage_dir)
            if runs:
                best_run = None
                best_improvement = -float("inf")
                best_value_loss = float("inf")

                for run_name, _ in runs.items():
                    analysis = analyze_run(f"{stage_dir}/{run_name}", args.agent)
                    for check in analysis.health_checks:
                        if check.name == "value_convergence":
                            if check.value > best_improvement:
                                best_improvement = check.value
                        if check.name == "value_loss_magnitude":
                            if check.value < best_value_loss:
                                best_value_loss = check.value

                stage_summaries.append(
                    {
                        "stage": stage,
                        "runs": len(runs),
                        "value_loss": best_value_loss,
                        "improvement": best_improvement,
                    }
                )

        # Print summary table
        print(f"{'=' * 80}")
        print("Summary by Stage")
        print(f"{'=' * 80}")
        print(f"{'Stage':<15} {'Runs':>6} {'Value Loss':>12} {'Improvement':>12}")
        print("-" * 46)

        for summary in stage_summaries:
            print(
                f"{summary['stage']:<15} {summary['runs']:>6} "
                f"{summary['value_loss']:>11.1f} {summary['improvement'] * 100:>11.1f}%"
            )

        print()


if __name__ == "__main__":
    main()
