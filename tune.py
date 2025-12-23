"""Hyperparameter tuning with Optuna for SAC on Humanoid-v5."""

import argparse
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from agents.wrappers import VelocityRewardWrapper

TUNING_DIR = Path("tuning")
N_ENVS = 32
N_EVAL_EPISODES = 5
EVAL_FREQ = 40_000
TOTAL_TIMESTEPS = 3_000_000

NET_ARCH_OPTIONS = {
    "small": [64, 64],
    "medium": [256, 256],
    "large": [400, 300],
}


def make_env():
    env = gym.make("Humanoid-v5")
    env = VelocityRewardWrapper(env, velocity_bonus=1.0)
    env = Monitor(env)
    return env


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

    # Suggest hyperparameters
    net_arch_type = trial.suggest_categorical("net_arch", list(NET_ARCH_OPTIONS.keys()))
    net_arch = NET_ARCH_OPTIONS[net_arch_type]
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    # Create training env
    train_env = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    # Create eval env (single env, no normalization updates)
    eval_env = SubprocVecEnv([make_env for _ in range(1)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    # Sync normalization stats from training env
    eval_env.obs_rms = train_env.obs_rms

    # Create model with TensorBoard logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_dir = TUNING_DIR / f"trial_{trial.number:03d}_{timestamp}"
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        policy_kwargs=dict(net_arch=dict(pi=net_arch, qf=net_arch)),
        tensorboard_log=str(trial_dir),
        verbose=0,
    )

    # Create callback for evaluation and pruning
    eval_callback = TrialEvalCallback(
        trial=trial,
        eval_env=eval_env,
        eval_freq=EVAL_FREQ // N_ENVS,  # Adjust for vectorized env
        n_eval_episodes=N_EVAL_EPISODES,
    )

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    except AssertionError:
        # Handle early termination from pruning
        pass
    finally:
        train_env.close()
        eval_env.close()

    if eval_callback.is_pruned:
        raise optuna.TrialPruned()

    return eval_callback.last_mean_reward


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for SAC")
    parser.add_argument("--n-trials", type=int, default=25, help="Number of trials")
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    parser.add_argument("--study-name", type=str, default="sac_humanoid", help="Study name")
    args = parser.parse_args()

    TUNING_DIR.mkdir(exist_ok=True)
    storage = f"sqlite:///{TUNING_DIR}/optuna.db"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5),
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
