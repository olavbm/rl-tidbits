"""Hyperparameter tuning with Optuna for SAC on Humanoid-v5."""

import argparse
import hashlib
from datetime import datetime
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from torch.utils.tensorboard import SummaryWriter

from agents.training import create_eval_env, create_sac_model, create_training_env

TUNING_DIR = Path("tuning")
N_ENVS = 16
N_EVAL_EPISODES = 4
EVAL_FREQ = 20_000      # doubled (gradient_steps=2 makes training slower)
TOTAL_TIMESTEPS = 2_000_000  # reduced for faster iteration
SAVE_FREQ = 100_000

BEST_LEARNING_RATE = 7.183183678075811e-05

NET_ARCH_OPTIONS = {
    "small": [64]*2,
    "deep": [64]*4,
    "medium": [256]*4,
    "large": [512]*3,
    "large_deep": [512]*5
}


class TrialEvalCallback(EvalCallback):
    """Callback for evaluating and reporting to Optuna for pruning."""

    def __init__(self, trial: optuna.Trial, eval_env, eval_freq: int, n_eval_episodes: int):
        super().__init__(
            eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            verbose=0,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        result = super()._on_step()

        if self.n_calls % self.eval_freq == 0:
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)

            if self.trial.should_prune():
                self.is_pruned = True
                return False

        return result


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for hyperparameter optimization."""
    study_id = trial.study.user_attrs["study_id"]
    trial.set_user_attr("study_id", study_id)  # Tag trial for filtering

    # Suggest hyperparameters
    net_arch_type = trial.suggest_categorical("net_arch", list(NET_ARCH_OPTIONS.keys()))
    net_arch = NET_ARCH_OPTIONS[net_arch_type]
    ent_coef_type = trial.suggest_categorical("ent_coef_type", ["auto", "fixed"])

    if ent_coef_type == "auto":
        ent_coef = "auto"
        # Search target entropy: default is -17 (action_dim), try higher values
        target_entropy = trial.suggest_float("target_entropy", -16.0, -4.0)
    else:
        # Fixed entropy coefficient, no auto-tuning
        ent_coef = trial.suggest_float("ent_coef", 0.05, 0.5, log=True)
        target_entropy = "auto"  # Ignored when ent_coef is fixed

    # Create environments
    train_env = create_training_env(n_envs=N_ENVS)
    eval_env = create_eval_env(train_env=train_env)

    # Create model with TensorBoard logging (nested directory structure)
    study_dir = TUNING_DIR / study_id
    study_dir.mkdir(parents=True, exist_ok=True)
    trial_dir = study_dir / f"trial_{trial.number:03d}"

    model = create_sac_model(
        env=train_env,
        learning_rate=BEST_LEARNING_RATE,
        net_arch=net_arch,
        tensorboard_log=trial_dir,
        ent_coef=ent_coef,
        target_entropy=target_entropy,
    )

    # Log hyperparameters to TensorBoard
    hparams = {
        "study_id": study_id,
        "trial": trial.number,
        "net_arch": net_arch_type,
        "ent_coef_type": ent_coef_type,
        "ent_coef": ent_coef if isinstance(ent_coef, float) else 0.0,
        "target_entropy": target_entropy if isinstance(target_entropy, float) else 0.0,
        "learning_rate": BEST_LEARNING_RATE,
    }

    # Create callback for evaluation and pruning
    eval_callback = TrialEvalCallback(
        trial=trial,
        eval_env=eval_env,
        eval_freq=EVAL_FREQ // N_ENVS,  # Adjust for vectorized env
        n_eval_episodes=N_EVAL_EPISODES,
    )

    # Checkpoint every 500k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ // N_ENVS,
        save_path=str(trial_dir),
        name_prefix="checkpoint",
        save_vecnormalize=True,
    )

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_callback, checkpoint_callback])
    except AssertionError:
        # Handle early termination from pruning
        pass
    finally:
        train_env.close()
        eval_env.close()
        train_env.close()
        eval_env.close()
        del model
        del train_env
        del eval_env
        import gc
        gc.collect()

    if eval_callback.is_pruned:
        # Log hparams for pruned trial too
        writer = SummaryWriter(log_dir=str(trial_dir / "SAC_1"))
        writer.add_hparams(hparams, {"hparam/reward": eval_callback.last_mean_reward})
        writer.close()
        raise optuna.TrialPruned()

    # Log final hparams with reward to TensorBoard
    writer = SummaryWriter(log_dir=str(trial_dir / "SAC_1"))
    writer.add_hparams(hparams, {"hparam/reward": eval_callback.last_mean_reward})
    writer.close()

    return eval_callback.last_mean_reward


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for SAC")
    parser.add_argument("--n-trials", type=int, default=25, help="Number of trials")
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    parser.add_argument("--study-name", type=str, default="sac_humanoid_sprint", help="Study name")
    args = parser.parse_args()

    TUNING_DIR.mkdir(exist_ok=True)
    storage = f"sqlite:///{TUNING_DIR}/optuna.db"

    # Generate unique study ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = hashlib.md5(f"{args.study_name}_{timestamp}".encode()).hexdigest()[:6]
    study_id = f"{timestamp}_{short_hash}"

    # Use unique study name per run to avoid pruning against old trials with different reward params
    unique_study_name = f"{args.study_name}_{study_id}"
    study = optuna.create_study(
        study_name=unique_study_name,
        storage=storage,
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.set_user_attr("study_id", study_id)

    print(f"Starting study: {args.study_name}")
    print(f"Study ID: {study_id}")
    print(f"Trials: {args.n_trials}")
    print(f"Storage: {storage}")
    print()

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True, n_jobs=4)

    # Find best trial from THIS run only (filter by study_id in user_attrs)
    current_run_trials = [
        t for t in study.trials
        if t.user_attrs.get("study_id") == study_id and t.value is not None
    ]
    if current_run_trials:
        best_current = max(current_run_trials, key=lambda t: t.value)
    else:
        best_current = study.best_trial  # fallback

    print("\n" + "=" * 50)
    print(f"Tuning complete. Study ID: {study_id}")
    print(f"Best trial (this run): {best_current.number}")
    print(f"  Value (mean reward): {best_current.value:.2f}")
    print("  Params:")
    for key, value in best_current.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
