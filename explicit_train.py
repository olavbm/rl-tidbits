"""Train SAC on Humanoid-v5 with best hyperparameters from Optuna trial 4."""

import argparse
from datetime import datetime
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from agents.training import create_eval_env, create_sac_model, create_training_env

RUNS_DIR = Path("runs")

# Best params from Optuna trial 4
BEST_NET_ARCH = [256, 256]  # medium (3-layer)
BEST_LEARNING_RATE = 7.183183678075811e-05


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


def main():
    parser = argparse.ArgumentParser(description="Train SAC with best hyperparameters")
    parser.add_argument("--steps", type=int, default=10_000_000, help="Total timesteps")
    parser.add_argument("--n-envs", type=int, default=16, help="Number of parallel envs")
    parser.add_argument(
        "--fix",
        type=str,
        choices=["high_target", "fixed"],
        help="Entropy fix: 'high_target' (target_entropy=-8) or 'fixed' (ent_coef=0.2)",
    )
    args = parser.parse_args()

    train(total_timesteps=args.steps, n_envs=args.n_envs, entropy_fix=args.fix)


if __name__ == "__main__":
    main()
