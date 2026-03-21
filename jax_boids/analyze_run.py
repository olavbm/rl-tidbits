"""Convergence analysis for IPPO training runs.

Reads TensorBoard logs and produces a text summary answering:
"Has training converged? Should I train longer?"

Usage:
    python -m jax_boids.analyze_run runs/train_best_pred2
    python -m jax_boids.analyze_run runs/train_best_pred2 --all
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from jax_boids.telemetry.reader import read_run


@dataclass
class MetricSummary:
    """Summary statistics for a single metric."""

    tag: str
    label: str
    early: float  # first 10% mean
    late: float  # last 10% mean
    last_1pct: float  # last 1% mean
    slope_last_10pct: float  # normalized slope over last 10%
    trend: str  # "decreasing", "plateaued", "increasing"
    status: str  # human-readable status label
    warning: bool = False


def find_latest_run(base_dir: str) -> Path:
    """Find the latest timestamped run directory.

    Looks for ippo_*, pred_*, predator_*, prey_* subdirectories.
    Falls back to base_dir if it contains event files directly.
    """
    base = Path(base_dir).resolve()
    prefixes = ("ippo_", "pred_", "predator_", "prey_")

    subdirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(prefixes)]
    if subdirs:
        return max(subdirs, key=lambda d: d.name)

    # Check if base_dir itself has event files
    events = list(base.glob("events.out.tfevents.*"))
    if events:
        return base

    raise FileNotFoundError(f"No TensorBoard runs found in {base_dir}")


def compute_normalized_slope(values: np.ndarray) -> float:
    """Compute slope of values normalized by the series range.

    Returns a value where |slope| < 0.01 means effectively flat.
    """
    if len(values) < 2:
        return 0.0

    x = np.arange(len(values), dtype=np.float64)
    y = values.astype(np.float64)
    slope, _ = np.polyfit(x, y, 1)

    value_range = float(y.max() - y.min())
    if value_range < 1e-10:
        return 0.0

    # Normalize: slope * n_points / range gives relative change over the window
    return float(slope * len(values) / value_range)


def classify_trend(norm_slope: float, threshold: float = 0.05) -> str:
    """Classify a normalized slope as decreasing/plateaued/increasing."""
    if norm_slope < -threshold:
        return "decreasing"
    elif norm_slope > threshold:
        return "increasing"
    return "plateaued"


def analyze_metric(
    values: pd.Series,
    tag: str,
    label: str,
) -> MetricSummary:
    """Analyze a single metric series."""
    v = values.to_numpy(dtype=np.float64)
    n = len(v)

    first_n = max(10, n // 10)
    last_n = max(10, n // 10)
    last_1pct_n = max(3, n // 100)

    early = float(v[:first_n].mean())
    late = float(v[-last_n:].mean())
    last_1pct = float(v[-last_1pct_n:].mean())

    tail = v[-last_n:]
    norm_slope = compute_normalized_slope(tail)
    trend = classify_trend(norm_slope)

    return MetricSummary(
        tag=tag,
        label=label,
        early=early,
        late=late,
        last_1pct=last_1pct,
        slope_last_10pct=norm_slope,
        trend=trend,
        status="",
    )


# ---------------------------------------------------------------------------
# KL diagnostics
# Thresholds from PPO literature (Schulman 2017, Spinning Up, CleanRL):
#   healthy: 0.005–0.02, acceptable: 0.02–0.05, unstable: >0.05
# ---------------------------------------------------------------------------

def classify_kl(summary: MetricSummary) -> MetricSummary:
    """Classify KL divergence health."""
    late = summary.late
    if late < 0.005:
        summary.status = "too low (policy barely updating)"
        summary.warning = True
    elif late <= 0.02:
        summary.status = "healthy"
    elif late <= 0.05:
        summary.status = "high but acceptable"
    else:
        summary.status = "unstable"
        summary.warning = True
    return summary


# ---------------------------------------------------------------------------
# Entropy diagnostics
# For continuous (Gaussian) actions, entropy can go negative — that's normal.
# We track relative drop from initial entropy instead of absolute thresholds.
#   >90% drop from early = collapsed, >70% = low
# ---------------------------------------------------------------------------

def classify_entropy(summary: MetricSummary) -> MetricSummary:
    """Classify entropy health based on relative drop from early training."""
    if abs(summary.early) < 1e-8:
        summary.status = "no initial entropy"
        summary.warning = True
        return summary

    # Use absolute values for drop calculation since entropy can be negative
    # for continuous policies. What matters is the magnitude shrinking.
    drop = (summary.early - summary.late) / abs(summary.early)
    if drop > 0.9:
        summary.status = f"collapsed (dropped {drop:.0%})"
        summary.warning = True
    elif drop > 0.7:
        summary.status = f"low (dropped {drop:.0%})"
        summary.warning = True
    else:
        summary.status = "healthy"
    return summary


def classify_reward_or_alive(summary: MetricSummary) -> MetricSummary:
    """Classify reward or prey_alive — just uses the trend."""
    summary.status = summary.trend
    return summary


# ---------------------------------------------------------------------------
# Verdict logic
# Priority: degenerate > unstable > still learning > converged
# ---------------------------------------------------------------------------

Verdict = Literal[
    "DEGENERATE",
    "UNSTABLE",
    "STILL LEARNING",
    "CONVERGED",
    "UNCLEAR",
]


def compute_verdict(summaries: dict[str, MetricSummary]) -> tuple[Verdict, str]:
    """Determine overall training verdict from metric summaries."""
    # Check entropy collapse
    for key in ("pred_entropy", "prey_entropy"):
        s = summaries.get(key)
        if s and "collapsed" in s.status:
            return "DEGENERATE", f"{s.label} entropy collapsed"

    # Check KL instability
    for key in ("pred_kl", "prey_kl"):
        s = summaries.get(key)
        if s and s.status == "unstable":
            return "UNSTABLE", f"{s.label} KL too high (late={s.late:.4f})"

    # Check if still learning
    alive = summaries.get("prey_alive")
    if alive and alive.trend == "decreasing":
        return "STILL LEARNING", "prey_alive still decreasing, train longer"

    pred_reward = summaries.get("pred_reward")
    if pred_reward and pred_reward.trend == "increasing":
        return "STILL LEARNING", "predator reward still increasing, train longer"

    prey_reward = summaries.get("prey_reward")
    if prey_reward and prey_reward.trend == "decreasing":
        # Prey reward decreasing means predators are getting better
        return "STILL LEARNING", "prey reward still decreasing, train longer"

    # Check convergence
    if alive and alive.trend == "plateaued":
        return "CONVERGED", "prey_alive flat for last 10%"

    if pred_reward and pred_reward.trend == "plateaued":
        return "CONVERGED", "predator reward plateaued"

    return "UNCLEAR", "check metrics manually"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_val(v: float) -> str:
    """Format a metric value with adaptive precision."""
    if abs(v) >= 10:
        return f"{v:.1f}"
    elif abs(v) >= 1:
        return f"{v:.2f}"
    elif abs(v) >= 0.01:
        return f"{v:.3f}"
    return f"{v:.4f}"


def print_summary(
    run_name: str,
    total_steps: int,
    summaries: dict[str, MetricSummary],
    verdict: Verdict,
    verdict_reason: str,
) -> None:
    """Print the convergence analysis."""
    print(f"\n=== Convergence Analysis: {run_name} ===")
    print(f"Steps: {total_steps:,}")
    print()

    # Metric lines
    for s in summaries.values():
        label = f"{s.label}:"
        warn = " !!" if s.warning else ""

        if s.tag in ("env/prey_alive", "pred/reward", "prey/reward",
                      "pred/value_loss", "prey/value_loss"):
            line = (
                f"  {label:<16} early={fmt_val(s.early):>8}  "
                f"late={fmt_val(s.late):>8}  "
                f"last_1%={fmt_val(s.last_1pct):>8}  "
                f"trend: {s.status}{warn}"
            )
        else:
            line = (
                f"  {label:<16} early={fmt_val(s.early):>8}  "
                f"late={fmt_val(s.late):>8}  "
                f"-- {s.status}{warn}"
            )
        print(line)

    print(f"\n=== Verdict: {verdict} ===")
    print(f"  {verdict_reason}")
    print()


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

METRIC_SPECS = [
    ("env/prey_alive", "prey_alive", "prey_alive", classify_reward_or_alive),
    ("pred/approx_kl", "pred_kl", "pred_kl", classify_kl),
    ("prey/approx_kl", "prey_kl", "prey_kl", classify_kl),
    ("pred/entropy", "pred_entropy", "pred_entropy", classify_entropy),
    ("prey/entropy", "prey_entropy", "prey_entropy", classify_entropy),
    ("pred/reward", "pred_reward", "pred_reward", classify_reward_or_alive),
    ("prey/reward", "prey_reward", "prey_reward", classify_reward_or_alive),
    ("pred/value_loss", "pred_vloss", "pred_vloss", classify_reward_or_alive),
    ("prey/value_loss", "prey_vloss", "prey_vloss", classify_reward_or_alive),
]


def analyze_single_run(run_dir: Path) -> None:
    """Analyze a single TensorBoard run directory."""
    metrics = read_run(str(run_dir))

    if not metrics:
        print(f"No metrics found in {run_dir}")
        return

    # Total steps
    total_steps = 0
    for df in metrics.values():
        if not df.empty:
            total_steps = max(total_steps, int(df["step"].max()))

    summaries: dict[str, MetricSummary] = {}

    for tag, key, label, classifier in METRIC_SPECS:
        if tag not in metrics:
            continue
        values = metrics[tag]["value"]
        if len(values) < 10:
            continue
        summary = analyze_metric(values, tag, label)
        summary = classifier(summary)
        summaries[key] = summary

    verdict, reason = compute_verdict(summaries)
    print_summary(run_dir.name, total_steps, summaries, verdict, reason)


def main():
    parser = argparse.ArgumentParser(
        description="Convergence analysis for IPPO training runs"
    )
    parser.add_argument(
        "run_dir",
        help="Path to run directory (auto-finds latest timestamped subdir)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all timestamped subdirs, not just latest",
    )
    args = parser.parse_args()

    base = Path(args.run_dir).resolve()

    if args.all:
        prefixes = ("ippo_", "pred_", "predator_", "prey_")
        subdirs = sorted(
            d for d in base.iterdir() if d.is_dir() and d.name.startswith(prefixes)
        )
        if not subdirs:
            print(f"No timestamped run directories found in {args.run_dir}")
            return
        for subdir in subdirs:
            analyze_single_run(subdir)
    else:
        run_dir = find_latest_run(args.run_dir)
        analyze_single_run(run_dir)


if __name__ == "__main__":
    main()
