"""Diagnostics and analysis for RL training metrics."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from jax_boids.telemetry.reader import read_run, read_all_runs


@dataclass
class HealthCheckResult:
    """Result of a single health check."""

    name: str
    status: Literal["OK", "WARNING", "CRITICAL"]
    value: float
    threshold: tuple[float, float | None] | None = None
    message: str = ""


@dataclass
class RunAnalysis:
    """Complete analysis of a training run."""

    run_name: str
    total_steps: int
    metrics_summary: dict[str, dict[str, Any]]
    health_checks: list[HealthCheckResult]
    learning_assessment: str


def compute_trend(values: pd.Series) -> dict[str, Any]:
    """Compute trend statistics for a metric series.

    Args:
        values: Metric values over time

    Returns:
        Dict with mean, std, min, max, first_10%, last_10%, trend
    """
    if len(values) < 2:
        return {"error": "Not enough data points"}

    values = values.dropna()
    if len(values) < 2:
        return {"error": "Not enough non-null data points"}

    n = len(values)
    first_n = max(10, n // 10)
    last_n = max(10, n // 10)

    # Compute trend (simple linear regression slope)
    x = np.arange(len(values))
    y = values.to_numpy(dtype=np.float64)
    slope, _ = np.polyfit(x, y, 1)

    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "first_10pct_mean": float(values.iloc[:first_n].mean()),
        "last_10pct_mean": float(values.iloc[-last_n:].mean()),
        "trend_slope": float(slope),
        "trend_direction": "decreasing" if slope < 0 else "increasing",
    }


def _get_float(stats: dict[str, Any], key: str) -> float:
    """Safely extract float from stats dict."""
    val = stats[key]
    if isinstance(val, str):
        return 0.0
    return float(val)


def check_value_convergence(df: pd.DataFrame) -> HealthCheckResult:
    """Check if value loss is decreasing (value function learning).

    Value loss should generally decrease over time as the value function
    learns to predict returns better.

    Args:
        df: DataFrame with 'value' column

    Returns:
        HealthCheckResult with convergence assessment
    """
    stats = compute_trend(df["value"])  # type: ignore[arg-type]
    if "error" in stats:
        return HealthCheckResult(
            name="value_convergence",
            status="WARNING",
            value=0.0,
            message="Insufficient data",
        )

    first_mean = _get_float(stats, "first_10pct_mean")
    last_mean = _get_float(stats, "last_10pct_mean")
    improvement = (first_mean - last_mean) / (first_mean + 1e-8)

    if improvement > 0.3:
        status = "OK"
        message = f"Value loss improved {improvement * 100:.1f}%"
    elif improvement > 0:
        status = "WARNING"
        message = f"Slow improvement: {improvement * 100:.1f}%"
    else:
        status = "CRITICAL"
        message = f"Value loss INCREASING: {-improvement * 100:.1f}%"

    return HealthCheckResult(
        name="value_convergence",
        status=status,
        value=improvement,
        threshold=(0.0, 0.3),
        message=message,
    )


def check_policy_stability(df: pd.DataFrame) -> HealthCheckResult:
    """Check if KL divergence is bounded (stable policy updates).

    Approx KL should stay in reasonable range (0.01-0.1 typical for PPO).
    Too high = unstable updates, too low = no learning.

    Args:
        df: DataFrame with 'value' column (approx_kl values)

    Returns:
        HealthCheckResult with stability assessment
    """
    stats = compute_trend(df["value"])  # type: ignore[arg-type]
    if "error" in stats:
        return HealthCheckResult(
            name="policy_stability",
            status="WARNING",
            value=0.0,
            message="Insufficient data",
        )

    mean_kl = _get_float(stats, "mean")
    max_kl = _get_float(stats, "max")

    if 0.01 <= mean_kl <= 0.1 and max_kl < 0.3:
        status = "OK"
        message = f"KL well-bounded (mean={mean_kl:.4f}, max={max_kl:.4f})"
    elif mean_kl < 0.01:
        status = "WARNING"
        message = f"KL very low - policy not updating (mean={mean_kl:.4f})"
    elif max_kl > 0.5:
        status = "CRITICAL"
        message = f"KL too high - unstable updates (max={max_kl:.4f})"
    else:
        status = "WARNING"
        message = f"KL borderline (mean={mean_kl:.4f}, max={max_kl:.4f})"

    return HealthCheckResult(
        name="policy_stability",
        status=status,
        value=mean_kl,
        threshold=(0.01, 0.1),
        message=message,
    )


def check_entropy_health(df: pd.DataFrame) -> HealthCheckResult:
    """Check if entropy is healthy (not collapsed).

    Entropy should decrease slowly but not collapse to near-zero
    (which would mean no exploration).

    Args:
        df: DataFrame with 'value' column (entropy values)

    Returns:
        HealthCheckResult with entropy health assessment
    """
    stats = compute_trend(df["value"])  # type: ignore[arg-type]
    if "error" in stats:
        return HealthCheckResult(
            name="entropy_health",
            status="WARNING",
            value=0.0,
            message="Insufficient data",
        )

    mean_entropy = _get_float(stats, "mean")
    last_entropy = _get_float(stats, "last_10pct_mean")
    first_entropy = _get_float(stats, "first_10pct_mean")

    if last_entropy < 0.1:
        status = "CRITICAL"
        message = f"Entropy collapsed! (last={last_entropy:.4f})"
    elif last_entropy < 0.5:
        status = "WARNING"
        message = f"Entropy low (last={last_entropy:.4f})"
    elif last_entropy < first_entropy * 0.5:
        status = "WARNING"
        message = (
            f"Entropy dropped significantly (first={first_entropy:.2f} -> last={last_entropy:.2f})"
        )
    else:
        status = "OK"
        message = f"Entropy healthy (mean={mean_entropy:.4f}, last={last_entropy:.4f})"

    return HealthCheckResult(
        name="entropy_health",
        status=status,
        value=mean_entropy,
        threshold=(0.5, None),
        message=message,
    )


def check_policy_loss(df: pd.DataFrame) -> HealthCheckResult:
    """Check policy loss magnitude.

    Policy loss should be near zero with small updates.
    Large positive values indicate the policy is being pushed hard.

    Args:
        df: DataFrame with 'value' column (policy_loss values)

    Returns:
        HealthCheckResult with policy loss assessment
    """
    stats = compute_trend(df["value"])  # type: ignore[arg-type]
    if "error" in stats:
        return HealthCheckResult(
            name="policy_loss",
            status="WARNING",
            value=0.0,
            message="Insufficient data",
        )

    mean_loss = _get_float(stats, "mean")
    abs_mean = abs(mean_loss)

    if abs_mean < 0.01:
        status = "OK"
        message = f"Policy loss near zero (mean={mean_loss:.6f})"
    elif abs_mean < 0.1:
        status = "OK"
        message = f"Policy loss reasonable (mean={mean_loss:.4f})"
    else:
        status = "WARNING"
        message = f"Policy loss high (mean={mean_loss:.4f})"

    return HealthCheckResult(
        name="policy_loss",
        status=status,
        value=mean_loss,
        threshold=(-0.1, 0.1),
        message=message,
    )


def check_value_loss_magnitude(df: pd.DataFrame) -> HealthCheckResult:
    """Check value loss absolute magnitude.

    Very high value loss indicates poor value function predictions.
    This is expected early in training but should decrease.

    Args:
        df: DataFrame with 'value' column (value_loss values)

    Returns:
        HealthCheckResult with value loss magnitude assessment
    """
    stats = compute_trend(df["value"])  # type: ignore[arg-type]
    if "error" in stats:
        return HealthCheckResult(
            name="value_loss_magnitude",
            status="WARNING",
            value=0.0,
            message="Insufficient data",
        )

    mean_loss = _get_float(stats, "mean")
    last_loss = _get_float(stats, "last_10pct_mean")

    if last_loss < 100:
        status = "OK"
        message = f"Value loss low (mean={mean_loss:.1f}, last={last_loss:.1f})"
    elif last_loss < 500:
        status = "OK"
        message = f"Value loss moderate (mean={mean_loss:.1f}, last={last_loss:.1f})"
    elif last_loss < 1000:
        status = "WARNING"
        message = f"Value loss high (mean={mean_loss:.1f}, last={last_loss:.1f})"
    else:
        status = "WARNING"
        message = f"Value loss very high (mean={mean_loss:.1f}, last={last_loss:.1f})"

    return HealthCheckResult(
        name="value_loss_magnitude",
        status=status,
        value=last_loss,
        threshold=(0.0, 500.0),
        message=message,
    )


def analyze_run(log_dir: str, agent: str = "pred") -> RunAnalysis:
    """Analyze a single training run.

    Args:
        log_dir: Path to TensorBoard log directory
        agent: Agent prefix ("pred" or "prey")

    Returns:
        RunAnalysis with all diagnostics
    """
    metrics = read_run(log_dir)

    if not metrics:
        return RunAnalysis(
            run_name=log_dir,
            total_steps=0,
            metrics_summary={},
            health_checks=[],
            learning_assessment="No metrics found",
        )

    # Get metric names
    value_tag = f"{agent}/value_loss"
    kl_tag = f"{agent}/approx_kl"
    entropy_tag = f"{agent}/entropy"
    policy_tag = f"{agent}/policy_loss"
    reward_tag = f"{agent}/reward"

    # Compute summaries
    summaries: dict[str, dict[str, Any]] = {}
    for tag in [value_tag, kl_tag, entropy_tag, policy_tag, reward_tag]:
        if tag in metrics:
            summaries[tag] = compute_trend(metrics[tag]["value"])  # type: ignore[arg-type]

    # Run health checks
    checks: list[HealthCheckResult] = []

    if value_tag in metrics:
        checks.append(check_value_convergence(metrics[value_tag]))
        checks.append(check_value_loss_magnitude(metrics[value_tag]))
    if kl_tag in metrics:
        checks.append(check_policy_stability(metrics[kl_tag]))
    if entropy_tag in metrics:
        checks.append(check_entropy_health(metrics[entropy_tag]))
    if policy_tag in metrics:
        checks.append(check_policy_loss(metrics[policy_tag]))

    # Get total steps
    total_steps = 0
    for df in metrics.values():
        if not df.empty:
            total_steps = max(total_steps, int(df["step"].max()))

    # Generate learning assessment
    critical_count = sum(1 for c in checks if c.status == "CRITICAL")
    warning_count = sum(1 for c in checks if c.status == "WARNING")

    if critical_count > 0:
        assessment = f"CRITICAL: {critical_count} critical issues found"
    elif warning_count > 1:
        assessment = f"WARNING: {warning_count} warnings, monitor closely"
    elif warning_count == 1:
        assessment = "OK: Minor issues, overall healthy"
    else:
        assessment = "HEALTHY: All checks passed"

    return RunAnalysis(
        run_name=log_dir,
        total_steps=total_steps,
        metrics_summary=summaries,
        health_checks=checks,
        learning_assessment=assessment,
    )


def print_analysis(analysis: RunAnalysis, verbose: bool = False) -> None:
    """Print analysis results to console.

    Args:
        analysis: RunAnalysis object
        verbose: Print detailed metrics
    """
    print(f"\n{'=' * 60}")
    print(f"Run: {analysis.run_name}")
    print(f"Steps: {analysis.total_steps:,}")
    print(f"Assessment: {analysis.learning_assessment}")
    print(f"{'=' * 60}")

    print("\nHealth Checks:")
    print("-" * 40)
    for check in analysis.health_checks:
        status_icon = {"OK": "✓", "WARNING": "⚠", "CRITICAL": "✗"}
        icon = status_icon.get(check.status, "?")
        print(f"  {icon} {check.name}: {check.status}")
        print(f"      {check.message}")

    if verbose:
        print("\nMetric Summaries:")
        print("-" * 40)
        for tag, stats in analysis.metrics_summary.items():
            print(f"\n  {tag}:")
            for key, value in stats.items():
                if key in ["mean", "std", "first_10pct_mean", "last_10pct_mean"]:
                    print(f"    {key}: {value:.6f}")
                elif key == "trend_direction":
                    print(f"    {key}: {value}")


def compare_runs(base_dir: str, agent: str = "pred") -> None:
    """Compare multiple runs in a directory.

    Args:
        base_dir: Directory containing multiple run subdirectories
        agent: Agent prefix ("pred" or "prey")
    """
    runs = read_all_runs(base_dir)

    if not runs:
        print(f"No runs found in {base_dir}")
        return

    print(f"\n{'=' * 60}")
    print(f"Comparing {len(runs)} runs in {base_dir}")
    print(f"{'=' * 60}")

    analyses = []
    for run_name, metrics in runs.items():
        analysis = analyze_run(f"{base_dir}/{run_name}", agent)
        analyses.append(analysis)

    # Print summary table
    print(f"\n{'Run':<30} {'Steps':>12} {'Assessment':<25}")
    print("-" * 67)

    for analysis in analyses:
        run_name = analysis.run_name.split("/")[-1][:28]
        print(f"{run_name:<30} {analysis.total_steps:>12,} {analysis.learning_assessment[:23]:<25}")

    # Print detailed analysis for each run
    for analysis in analyses:
        print_analysis(analysis, verbose=False)


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze TensorBoard training logs")
    parser.add_argument("log_dir", help="Path to log directory or run directory")
    parser.add_argument(
        "--agent", default="pred", choices=["pred", "prey"], help="Agent to analyze"
    )
    parser.add_argument("--compare", action="store_true", help="Compare all runs in directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.compare:
        compare_runs(args.log_dir, args.agent)
    else:
        analysis = analyze_run(args.log_dir, args.agent)
        print_analysis(analysis, verbose=args.verbose)


if __name__ == "__main__":
    main()
