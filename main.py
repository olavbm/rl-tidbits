import argparse
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from agents.wrappers import StandingRewardWrapper

RUNS_DIR = Path("runs")


def make_env():
    env = gym.make("Humanoid-v5")
    env = StandingRewardWrapper(env, velocity_penalty=5.0)
    env = Monitor(env)  # Track episode rewards and lengths
    return env


def train(total_timesteps: int = 1_000_000):
    # Create unique run directory with timestamp
    run_name = f"SAC_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Vectorized env with normalization (critical for performance!)
    n_envs = 12  # Match performance core count
    env = SubprocVecEnv([make_env for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # SAC with TensorBoard logging (logs go to run_dir)
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(run_dir),
        device="cuda"
    )

    # Save checkpoints every 50k steps (also to run_dir)
    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,
        save_path=str(run_dir),
        name_prefix="checkpoint",
        save_vecnormalize=True,
    )

    # Train
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # Save final model + normalizer stats
    model.save(run_dir / "final")
    env.save(run_dir / "final_vecnormalize.pkl")
    print(f"Final model saved to {run_dir}")


def evaluate(checkpoint: str | None = None):
    # Find checkpoint and normalizer paths
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        # Strip extensions user might have included
        name = checkpoint_path.name.removesuffix(".zip")
        checkpoint_path = checkpoint_path.parent / name

        # CheckpointCallback naming: checkpoint_500_steps.zip -> checkpoint_vecnormalize_500_steps.pkl
        # Extract prefix and steps from "checkpoint_500_steps"
        parts = name.rsplit("_", 2)  # ['checkpoint', '500', 'steps']
        if len(parts) == 3:
            normalizer_name = f"{parts[0]}_vecnormalize_{parts[1]}_{parts[2]}.pkl"
        else:
            normalizer_name = f"{name}_vecnormalize.pkl"  # fallback for 'final'
        normalizer_path = checkpoint_path.parent / normalizer_name
    else:
        # Default to latest run's final model
        runs = sorted(RUNS_DIR.glob("SAC_*"))
        if not runs:
            print("No runs found. Train first with: python main.py")
            return
        latest_run = runs[-1]
        checkpoint_path = latest_run / "final"
        normalizer_path = latest_run / "final_vecnormalize.pkl"

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Loading normalizer: {normalizer_path}")

    # Load and evaluate with rendering
    def make_eval_env():
        env = gym.make("Humanoid-v5", render_mode="human")
        return StandingRewardWrapper(env, velocity_penalty=1.0)

    env = DummyVecEnv([make_eval_env])
    env = VecNormalize.load(normalizer_path, env)
    env.training = False  # Don't update stats during eval
    env.norm_reward = False

    model = SAC.load(checkpoint_path, env=env)

    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)


def main():
    parser = argparse.ArgumentParser(description="SAC for Humanoid-v5")
    parser.add_argument("--eval", action="store_true", help="Evaluate saved model")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for evaluation")
    args = parser.parse_args()

    if args.eval:
        evaluate(checkpoint=args.checkpoint)
    else:
        train(total_timesteps=args.steps)


if __name__ == "__main__":
    main()
