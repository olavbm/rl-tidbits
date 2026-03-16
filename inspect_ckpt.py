"""Inspect checkpoint parameters and training config."""

import json
import sys
import numpy as np
import orbax.checkpoint as ocp
from pathlib import Path

run_dir = Path(
    sys.argv[1] if len(sys.argv) > 1 else "runs/train_ippo_accel/ippo_20260316_094356"
).resolve()

# Config
with open(run_dir / "config.json") as f:
    cfg = json.load(f)

print("=== TRAIN CONFIG ===")
tc = cfg["train"]
for k in [
    "lr",
    "n_epochs",
    "n_envs",
    "total_timesteps",
    "ent_coef",
    "clip_eps",
    "n_steps",
    "n_minibatches",
    "prey_lr",
    "prey_ent_coef",
]:
    print(f"  {k}: {tc.get(k, 'N/A')}")

print("\n=== ENV CONFIG ===")
ec = cfg["env"]
for k in [
    "max_acceleration",
    "velocity_damping",
    "prey_speed_mult",
    "boids_strength",
    "max_speed",
    "predator_speed_bonus",
    "dt",
]:
    print(f"  {k}: {ec.get(k, 'N/A')}")

if "prey_hparams" in cfg:
    print(f"\n=== PREY HPARAMS ===")
    for k, v in cfg["prey_hparams"].items():
        print(f"  {k}: {v}")

# Use PyTreeCheckpointer like visualize_trained.py
checkpointer = ocp.PyTreeCheckpointer()


def inspect_params(name, params):
    print(f"\n=== {name.upper()} ===")
    logstd = np.array(params["params"]["action_logstd"])
    print(f"  action_logstd: {logstd}")
    print(f"  action_std:    {np.exp(logstd)}")

    # Find all Dense layers
    param_keys = sorted(params["params"].keys())
    print(f"  param keys: {param_keys}")

    dense_layers = [k for k in param_keys if k.startswith("Dense")]
    for layer in dense_layers:
        w = np.array(params["params"][layer]["kernel"])
        b = np.array(params["params"][layer]["bias"])
        print(
            f"  {layer}: kernel {w.shape} mean={w.mean():.4f} std={w.std():.4f} | bias mean={b.mean():.4f} range=[{b.min():.4f}, {b.max():.4f}]"
        )


pred_params = checkpointer.restore(str(run_dir / "checkpoint_pred"))
inspect_params("predator", pred_params)

prey_params = checkpointer.restore(str(run_dir / "checkpoint_prey"))
inspect_params("prey", prey_params)

# Compare: how much did params change?
# Check if prey logstd moved from initialization (0.0)
prey_logstd = np.array(prey_params["params"]["action_logstd"])
pred_logstd = np.array(pred_params["params"]["action_logstd"])
print(f"\n=== LEARNING DIAGNOSIS ===")
print(f"  Pred logstd moved from 0.0 by: {np.abs(pred_logstd).mean():.6f}")
print(f"  Prey logstd moved from 0.0 by: {np.abs(prey_logstd).mean():.6f}")
print(f"  (If prey logstd ~0.0, prey policy barely updated)")
