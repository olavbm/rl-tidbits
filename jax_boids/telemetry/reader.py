"""Read TensorBoard event files using tbparse."""

from pathlib import Path

import pandas as pd
from tbparse import SummaryReader


def read_run(log_dir: str) -> dict[str, pd.DataFrame]:
    """Read all scalar metrics from a TensorBoard run directory.

    Args:
        log_dir: Path to TensorBoard log directory (containing events files)

    Returns:
        Dictionary mapping tag names to DataFrames with columns:
        - step: Training step
        - wall_time: Timestamp
        - value: Metric value

    Example:
        ```python
        metrics = read_run("runs/exp1")
        df = metrics["pred/value_loss"]
        print(df.tail())
        ```
    """
    reader = SummaryReader(log_dir)
    df = reader.scalars

    if df.empty:
        return {}

    # Group by tag and return dict of DataFrames
    result = {}
    for tag, group in df.groupby("tag"):
        result[tag] = group[["step", "value"]].reset_index(drop=True)

    return result


def read_all_runs(base_dir: str) -> dict[str, dict[str, pd.DataFrame]]:
    """Read all runs in a directory (e.g., all stage1_easy runs).

    Args:
        base_dir: Parent directory containing multiple run subdirectories

    Returns:
        Dictionary mapping run names to their metrics dicts

    Example:
        ```python
        runs = read_all_runs("runs/curriculum/stage1_easy")
        for run_name, metrics in runs.items():
            print(run_name, len(metrics))
        ```
    """
    base_path = Path(base_dir)
    results = {}

    for entry in sorted(base_path.iterdir()):
        if entry.is_dir():
            metrics = read_run(str(entry))
            if metrics:  # Only include if we got data
                results[entry.name] = metrics

    return results


def get_metric_series(
    metrics: dict[str, pd.DataFrame], tag: str, column: str = "value"
) -> pd.Series:
    """Get a metric series by tag name.

    Args:
        metrics: Dict from read_run()
        tag: Metric tag name (e.g., "pred/value_loss")
        column: Column to return ("step", "wall_time", or "value")

    Returns:
        Series, raises KeyError if tag not found
    """
    return metrics[tag][column]  # type: ignore


def get_metric_df(metrics: dict[str, pd.DataFrame], tag: str) -> pd.DataFrame | None:
    """Get full DataFrame for a metric tag.

    Args:
        metrics: Dict from read_run()
        tag: Metric tag name

    Returns:
        DataFrame with step, wall_time, value columns or None
    """
    return metrics.get(tag)
