"""Hyperparameter tuning with Optuna for SAC on Humanoid-v5."""

import argparse
import random
from datetime import datetime
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from agents.training import create_eval_env, create_sac_model, create_training_env

TUNING_DIR = Path("tuning")
N_ENVS = 16
N_EVAL_EPISODES = 5
EVAL_FREQ = 50_000
TOTAL_TIMESTEPS = 10_000_000

# Fixed best params from previous tuning
BEST_NET_ARCH = [256, 256]
BEST_LEARNING_RATE = 7.183183678075811e-05


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

    # Suggest entropy hyperparameters
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

    # Create model with TensorBoard logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_dir = TUNING_DIR / f"trial_{random.randint(0, 1000):04d}_{trial.number:03d}_{timestamp}"
    model = create_sac_model(
        env=train_env,
        learning_rate=BEST_LEARNING_RATE,
        net_arch=BEST_NET_ARCH,
        tensorboard_log=trial_dir,
        ent_coef=ent_coef,
        target_entropy=target_entropy,
    )

    # Create callback for evaluation and pruning
    eval_callback = TrialEvalCallback(
        trial=trial,
        eval_env=eval_env,
        eval_freq=EVAL_FREQ // N_ENVS,  # Adjust for vectorized env
        n_eval_episodes=N_EVAL_EPISODES,
    )

    # Checkpoint every 500k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000 // N_ENVS,
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
        raise optuna.TrialPruned()

    return eval_callback.last_mean_reward


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for SAC")
    parser.add_argument("--n-trials", type=int, default=25, help="Number of trials")
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    parser.add_argument("--study-name", type=str, default="sac_humanoid_entropy", help="Study name")
    args = parser.parse_args()

    TUNING_DIR.mkdir(exist_ok=True)
    storage = f"sqlite:///{TUNING_DIR}/optuna.db"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=20),  # 25 * 40k = 1M min
        load_if_exists=True,  # Always resume if study exists
    )

    print(f"Starting optimization: {args.n_trials} trials")
    print(f"Storage: {storage}")

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print("\n" + "=" * 50)
    print("Best trial:")
    print(f"  Value (mean reward): {study.best_trial.value:.2f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
