#!/usr/bin/env python3
"""Convenient script to run curriculum training."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_boids.envs.types import EnvConfig
from jax_boids.train import TrainConfig
from jax_boids.train_curriculum import train_curriculum


def main():
    train_config = TrainConfig(
        lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=2.0,
        ent_coef=0.01,
        n_steps=128,
        n_epochs=8,
        n_minibatches=4,
        n_envs=32,
    )

    base_env_config = EnvConfig(
        separation_weight=1.5,
        alignment_weight=1.0,
        cohesion_weight=1.0,
        perception_radius=15.0,
        capture_radius=0.5,
        predator_speed_bonus=1.5,  # Increased from 1.2
        max_speed=5.0,
        max_acceleration=2.0,
        dt=0.1,
        k_nearest_same=0,
        k_nearest_enemy=5,
    )

    train_curriculum(
        train_config,
        base_env_config,
        seed=42,
        log_dir="runs/curriculum_tuned",
        save_dir="checkpoints/curriculum_tuned",
    )


if __name__ == "__main__":
    main()
