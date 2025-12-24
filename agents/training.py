"""Shared training utilities for SAC on Humanoid-v5."""

from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from agents.wrappers import SprintRewardWrapper, VelocityRewardWrapper

# PyTorch performance optimizations
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")


def make_env(render_mode: str | None = None, sprint: bool = True):
    """Create a single Humanoid-v5 environment with reward wrapper.

    Args:
        sprint: If True, use aggressive speed reward. If False, use gentle velocity bonus.
    """
    env = gym.make("Humanoid-v5", render_mode=render_mode)
    if sprint:
        env = SprintRewardWrapper(env, speed_weight=5.0, survive_weight=0.5)
    else:
        env = VelocityRewardWrapper(env, velocity_bonus=1.0)
    env = Monitor(env)
    return env


def create_training_env(n_envs: int = 16):
    """Create vectorized training environment with normalization."""
    env = SubprocVecEnv([make_env for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    return env


def create_eval_env(train_env: VecNormalize | None = None):
    """Create evaluation environment, optionally syncing normalization stats."""
    env = SubprocVecEnv([make_env for _ in range(1)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
    if train_env is not None:
        env.obs_rms = train_env.obs_rms
    return env


def create_render_env(normalizer_path: Path | str):
    """Create environment for visualization with loaded normalizer."""
    def make_render_env():
        env = gym.make("Humanoid-v5", render_mode="human")
        return VelocityRewardWrapper(env, velocity_bonus=1.0)

    env = DummyVecEnv([make_render_env])
    env = VecNormalize.load(str(normalizer_path), env)
    env.training = False
    env.norm_reward = False
    return env


def create_sac_model(
    env,
    learning_rate: float,
    net_arch: list[int],
    tensorboard_log: str | Path | None = None,
    buffer_size: int = 400_000,
    batch_size: int = 10_000,
    gradient_steps: int = 1,
    train_freq: int = 8,
    device: str = "cuda",
    verbose: int = 0,
    compile_policy: bool = True,
    ent_coef: str | float = "auto",
    target_entropy: str | float = "auto",
):
    """Create SAC model with standard configuration.

    Args:
        ent_coef: Entropy coefficient. "auto" for auto-tuning, or a float for fixed value.
        target_entropy: Target entropy for auto-tuning. "auto" = -action_dim, or a float.
    """
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        policy_kwargs=dict(net_arch=dict(pi=net_arch, qf=net_arch)),
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gradient_steps=gradient_steps,
        train_freq=train_freq,
        device=device,
        verbose=verbose,
        ent_coef=ent_coef,
        target_entropy=target_entropy,
    )

    if compile_policy:
        model.policy = torch.compile(model.policy)

    return model
