"""Train SAC on Humanoid-v5 with best hyperparameters from Optuna trial 4."""

import argparse
import io
import json
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from agents.training import create_eval_env, create_sac_model, create_training_env, make_env

RUNS_DIR = Path("runs")
TUNING_DIR = Path("tuning")

# Best params from Optuna trial 4
BEST_NET_ARCH = [256, 256]  # medium (3-layer)
BEST_LEARNING_RATE = 7.183183678075811e-05


def find_checkpoint_and_normalizer(checkpoint_path: Path) -> tuple[Path, Path]:
    """Find checkpoint and matching normalizer from a checkpoint path."""
    checkpoint = checkpoint_path
    if not checkpoint.exists():
        checkpoint = Path(str(checkpoint_path) + ".zip")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Find normalizer
    trial_dir = checkpoint.parent
    name = checkpoint.stem
    parts = name.rsplit("_", 2)
    if len(parts) == 3:
        normalizer_name = f"{parts[0]}_vecnormalize_{parts[1]}_{parts[2]}.pkl"
    else:
        normalizer_name = f"{name}_vecnormalize.pkl"
    normalizer_path = trial_dir / normalizer_name

    if not normalizer_path.exists():
        normalizers = list(trial_dir.glob("*vecnormalize*.pkl"))
        if normalizers:
            normalizer_path = sorted(normalizers)[-1]
        else:
            raise FileNotFoundError(f"No normalizer found in {trial_dir}")

    return checkpoint, normalizer_path


def find_latest_checkpoint(trial_dir: Path) -> tuple[Path, Path]:
    """Find latest checkpoint and normalizer in a trial directory."""
    checkpoints = list(trial_dir.glob("checkpoint_*_steps.zip"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {trial_dir}")

    # Sort by step number
    def get_steps(p: Path) -> int:
        parts = p.stem.rsplit("_", 2)
        return int(parts[1]) if len(parts) >= 2 else 0

    latest = sorted(checkpoints, key=get_steps)[-1]
    return find_checkpoint_and_normalizer(latest)


def parse_resume_arg(resume_arg: str) -> tuple[Path, Path]:
    """Parse resume argument and return (checkpoint_path, normalizer_path).

    Formats:
    - Full path: path/to/checkpoint.zip
    - Study:trial: 20251225_135356_393a78:7
    - Trial dir: tuning/STUDY_ID/trial_NNN
    """
    path = Path(resume_arg)

    # If contains ':', parse as STUDY_ID:TRIAL_NUM (but not Windows paths like C:)
    if ":" in resume_arg and not path.exists() and len(resume_arg.split(":")[0]) > 1:
        study_id, trial_num = resume_arg.rsplit(":", 1)
        trial_dir = TUNING_DIR / study_id / f"trial_{int(trial_num):03d}"
        if not trial_dir.exists():
            raise FileNotFoundError(f"Trial directory not found: {trial_dir}")
        return find_latest_checkpoint(trial_dir)

    # If path to directory, find latest checkpoint
    if path.is_dir():
        return find_latest_checkpoint(path)

    # Otherwise treat as checkpoint path
    return find_checkpoint_and_normalizer(path)


def print_checkpoint_hyperparams(checkpoint: Path) -> None:
    """Print hyperparameters stored in a checkpoint."""
    with zipfile.ZipFile(str(checkpoint), "r") as zf:
        with zf.open("data") as f:
            data = json.load(f)
        with zf.open("pytorch_variables.pth") as f:
            pth_vars = torch.load(io.BytesIO(f.read()), weights_only=False, map_location="cpu")

    policy_kwargs = data.get("policy_kwargs", {})
    net_arch = policy_kwargs.get("net_arch", {})
    if isinstance(net_arch, dict):
        net_arch = net_arch.get("pi", net_arch.get("qf", []))

    print("\nLoaded hyperparameters:")
    print(f"  learning_rate: {data.get('learning_rate', 'N/A'):.2e}")
    print(f"  net_arch: {net_arch}")
    print(f"  ent_coef: {pth_vars.get('ent_coef_tensor', 'N/A'):.4f} (learned)")
    print(f"  timesteps: {data.get('num_timesteps', 'N/A'):,}")
    print()


def train(total_timesteps: int = 10_000_000, n_envs: int = 16, entropy_fix: str | None = None):
    """Train with best hyperparameters.

    Args:
        entropy_fix: None for default, "high_target" for target_entropy=-8,
                     "fixed" for ent_coef=0.2 (no auto-tuning)
    """
    # Configure entropy settings based on fix
    if entropy_fix == "high_target":
        ent_coef = "auto"
        target_entropy = -8.0  # Higher than default -17 (action_dim)
        run_suffix = "_high_target_entropy"
    elif entropy_fix == "fixed":
        ent_coef = 0.2  # Fixed, no auto-tuning
        target_entropy = "auto"
        run_suffix = "_fixed_ent_coef"
    else:
        ent_coef = "auto"
        target_entropy = "auto"
        run_suffix = ""

    run_name = f"best_SAC_{datetime.now().strftime('%Y%m%d_%H%M%S')}{run_suffix}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    print(f"Net arch: {BEST_NET_ARCH}")
    print(f"Learning rate: {BEST_LEARNING_RATE}")
    print(f"Entropy fix: {entropy_fix or 'none'}")
    print(f"  ent_coef: {ent_coef}")
    print(f"  target_entropy: {target_entropy}")

    env = create_training_env(n_envs=n_envs)
    eval_env = create_eval_env(train_env=env)

    model = create_sac_model(
        env=env,
        learning_rate=BEST_LEARNING_RATE,
        net_arch=BEST_NET_ARCH,
        tensorboard_log=run_dir,
        verbose=1,
        ent_coef=ent_coef,
        target_entropy=target_entropy,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=500_000 // n_envs,
        save_path=str(run_dir),
        name_prefix="checkpoint",
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=50_000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        log_path=str(run_dir),
    )

    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])

    model.save(run_dir / "final")
    env.save(run_dir / "final_vecnormalize.pkl")
    eval_env.close()
    print(f"Final model saved to {run_dir}")


def _fix_compiled_state_dict(state_dict: dict) -> dict:
    """Strip _orig_mod. prefix from torch.compile() saved checkpoints."""
    fixed = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "")
        fixed[new_key] = value
    return fixed


def resume(resume_arg: str, total_timesteps: int = 5_000_000, n_envs: int = 16):
    """Resume training from a checkpoint.

    Args:
        resume_arg: Checkpoint path, study:trial format (e.g., "20251225_135356_393a78:7"),
                    or trial directory path.
    """
    try:
        checkpoint, normalizer_path = parse_resume_arg(resume_arg)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Resuming from: {checkpoint}")
    print(f"Normalizer: {normalizer_path}")
    print_checkpoint_hyperparams(checkpoint)

    # Create new run directory for continued training
    run_name = f"resumed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_dir}")

    # Create base env WITHOUT VecNormalize wrapper, then load normalizer
    base_env = SubprocVecEnv([make_env for _ in range(n_envs)])
    env = VecNormalize.load(str(normalizer_path), base_env)
    env.training = True
    eval_env = create_eval_env(train_env=env)

    # Load model, fixing torch.compile keys if needed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    with zipfile.ZipFile(str(checkpoint), "r") as zf:
        with zf.open("policy.pth") as f:
            policy_state = torch.load(io.BytesIO(f.read()), weights_only=False, map_location=device)

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

        model = SAC.load(tmp_path, env=env, device=device)
    else:
        model = SAC.load(str(checkpoint.with_suffix("")), env=env, device=device)

    model.tensorboard_log = str(run_dir)

    checkpoint_callback = CheckpointCallback(
        save_freq=500_000 // n_envs,
        save_path=str(run_dir),
        name_prefix="checkpoint",
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=50_000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        log_path=str(run_dir),
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=False,
    )

    model.save(run_dir / "final")
    env.save(run_dir / "final_vecnormalize.pkl")
    eval_env.close()
    print(f"Final model saved to {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train SAC with best hyperparameters")
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume: path/to/checkpoint.zip, STUDY_ID:TRIAL_NUM, or trial_dir",
    )
    parser.add_argument("--steps", type=int, default=10_000_000, help="Total timesteps")
    parser.add_argument("--n-envs", type=int, default=16, help="Number of parallel envs")
    parser.add_argument(
        "--fix",
        type=str,
        choices=["high_target", "fixed"],
        help="Entropy fix: 'high_target' (target_entropy=-8) or 'fixed' (ent_coef=0.2)",
    )
    args = parser.parse_args()

    if args.resume:
        resume(args.resume, total_timesteps=args.steps, n_envs=args.n_envs)
    else:
        train(total_timesteps=args.steps, n_envs=args.n_envs, entropy_fix=args.fix)


if __name__ == "__main__":
    main()
