"""Telemetry and diagnostics for RL training runs."""

__all__ = ["read_run", "read_all_runs", "analyze_run", "compare_runs"]


def __getattr__(name):
    if name in ("read_run", "read_all_runs"):
        from jax_boids.telemetry.reader import read_all_runs, read_run

        return read_run if name == "read_run" else read_all_runs
    elif name in ("analyze_run", "compare_runs"):
        from jax_boids.telemetry.diagnostics import analyze_run, compare_runs

        return analyze_run if name == "analyze_run" else compare_runs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
