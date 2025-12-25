"""Evaluate and list tuning trials."""

import argparse
import io
import tempfile
import zipfile
from pathlib import Path

import torch
from stable_baselines3 import SAC

from agents.training import create_render_env

TUNING_DIR = Path("tuning")


def _fix_compiled_state_dict(state_dict: dict) -> dict:
    """Strip _orig_mod. prefix from torch.compile() saved checkpoints."""
    fixed = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "")
        fixed[new_key] = value
    return fixed


def list_trials(study_id: str):
    """List all trials for a study."""
    study_dir = TUNING_DIR / study_id
    if not study_dir.exists():
        print(f"Study not found: {study_dir}")
        return

    print(f"Study: {study_id}")
    print(f"Path: {study_dir}")
    print()

    for trial_dir in sorted(study_dir.glob("trial_*")):
        trial_num = trial_dir.name.split("_")[1]

        # Find checkpoints
        checkpoints = sorted(trial_dir.glob("checkpoint_*.zip"))
        checkpoint_info = f"{len(checkpoints)} checkpoints" if checkpoints else "no checkpoints"

        # Try to find latest checkpoint steps
        if checkpoints:
            latest = checkpoints[-1].stem
            steps = latest.split("_")[-2] + "_" + latest.split("_")[-1]
            checkpoint_info = f"latest: {steps}"

        print(f"  Trial {trial_num}: {checkpoint_info} | {trial_dir}")


def find_latest_checkpoint(trial_dir: Path) -> tuple[Path, Path] | None:
    """Find the latest checkpoint and normalizer in a trial directory."""
    checkpoints = sorted(trial_dir.glob("checkpoint_*.zip"))
    if not checkpoints:
        return None

    checkpoint = checkpoints[-1]
    name = checkpoint.stem

    # Find corresponding normalizer
    parts = name.rsplit("_", 2)
    if len(parts) == 3:
        normalizer_name = f"{parts[0]}_vecnormalize_{parts[1]}_{parts[2]}.pkl"
    else:
        normalizer_name = f"{name}_vecnormalize.pkl"

    normalizer = trial_dir / normalizer_name
    if not normalizer.exists():
        # Try alternative naming
        normalizers = list(trial_dir.glob("*vecnormalize*.pkl"))
        if normalizers:
            normalizer = normalizers[-1]
        else:
            return None

    return checkpoint, normalizer


def evaluate(study_id: str, trial_num: int):
    """Evaluate a specific trial with human rendering."""
    trial_dir = TUNING_DIR / study_id / f"trial_{trial_num:03d}"

    if not trial_dir.exists():
        print(f"Trial not found: {trial_dir}")
        return

    result = find_latest_checkpoint(trial_dir)
    if result is None:
        print(f"No checkpoint found in {trial_dir}")
        return

    checkpoint_path, normalizer_path = result

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Loading normalizer: {normalizer_path}")

    env = create_render_env(normalizer_path)

    # Load and fix compiled model state dict if needed
    checkpoint_file = str(checkpoint_path)

    with zipfile.ZipFile(checkpoint_file, "r") as zf:
        with zf.open("policy.pth") as f:
            policy_state = torch.load(io.BytesIO(f.read()), weights_only=False, map_location="cpu")

    # Check if policy state dict has _orig_mod. prefix from torch.compile()
    if any(k.startswith("_orig_mod.") for k in policy_state.keys()):
        print("Fixing torch.compile() checkpoint keys...")
        policy_state = _fix_compiled_state_dict(policy_state)

        # Repack the fixed checkpoint
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name

        with zipfile.ZipFile(checkpoint_file, "r") as zf_in:
            with zipfile.ZipFile(tmp_path, "w") as zf_out:
                for item in zf_in.namelist():
                    if item == "policy.pth":
                        buf = io.BytesIO()
                        torch.save(policy_state, buf)
                        zf_out.writestr("policy.pth", buf.getvalue())
                    else:
                        zf_out.writestr(item, zf_in.read(item))

        model = SAC.load(tmp_path, env=env, device="cpu")
    else:
        model = SAC.load(checkpoint_path.with_suffix(""), env=env, device="cpu")

    print("Running evaluation with rendering...")
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)


def evaluate_checkpoint(checkpoint_path: str):
    """Evaluate a checkpoint directly by path."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        # Try adding .zip
        checkpoint = Path(checkpoint_path + ".zip")
    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    trial_dir = checkpoint.parent

    # Find normalizer
    name = checkpoint.stem
    parts = name.rsplit("_", 2)
    if len(parts) == 3:
        normalizer_name = f"{parts[0]}_vecnormalize_{parts[1]}_{parts[2]}.pkl"
    else:
        normalizer_name = f"{name}_vecnormalize.pkl"

    normalizer = trial_dir / normalizer_name
    if not normalizer.exists():
        normalizers = list(trial_dir.glob("*vecnormalize*.pkl"))
        if normalizers:
            normalizer = normalizers[-1]
        else:
            print(f"No normalizer found in {trial_dir}")
            return

    print(f"Loading checkpoint: {checkpoint}")
    print(f"Loading normalizer: {normalizer}")

    env = create_render_env(normalizer)

    with zipfile.ZipFile(str(checkpoint), "r") as zf:
        with zf.open("policy.pth") as f:
            policy_state = torch.load(io.BytesIO(f.read()), weights_only=False, map_location="cpu")

    if any(k.startswith("_orig_mod.") for k in policy_state.keys()):
        print("Fixing torch.compile() checkpoint keys...")
        policy_state = _fix_compiled_state_dict(policy_state)

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name

        with zipfile.ZipFile(str(checkpoint), "r") as zf_in:
            with zipfile.ZipFile(tmp_path, "w") as zf_out:
                for item in zf_in.namelist():
                    if item == "policy.pth":
                        buf = io.BytesIO()
                        torch.save(policy_state, buf)
                        zf_out.writestr("policy.pth", buf.getvalue())
                    else:
                        zf_out.writestr(item, zf_in.read(item))

        model = SAC.load(tmp_path, env=env, device="cpu")
    else:
        model = SAC.load(str(checkpoint.with_suffix("")), env=env, device="cpu")

    print("Running evaluation with rendering...")
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)


def main():
    parser = argparse.ArgumentParser(description="Evaluate tuning trials")
    parser.add_argument("--list", type=str, metavar="STUDY_ID", help="List trials for a study")
    parser.add_argument("target", type=str, nargs="?", help="STUDY_ID:TRIAL_NUM or path to checkpoint")
    args = parser.parse_args()

    if args.list:
        list_trials(args.list)
    elif args.target:
        if ":" in args.target and not Path(args.target).exists():
            # Format: study_id:trial_num
            study_id, trial_num = args.target.rsplit(":", 1)
            evaluate(study_id, int(trial_num))
        else:
            # Direct path to checkpoint
            evaluate_checkpoint(args.target)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
