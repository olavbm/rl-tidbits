"""Curriculum training for single predator vs passive boids.

Trains a single predator to hunt passive boids (prey that only exhibit
boids flocking behavior, no learning). Uses curriculum learning with
4 stages of increasing difficulty.

Each stage trains for 2M timesteps, then advances to the next stage
with the learned policy preserved.
"""

import os

from flax import serialization

from jax_boids.envs.curriculum import TIMESTEPS_PER_STAGE, get_default_curriculum
from jax_boids.envs.types import EnvConfig
from jax_boids.train import TrainConfig, train


def save_predator_policy(pred_state, save_path: str):
    """Save predator policy to disk."""
    state_dict = serialization.to_state_dict(pred_state)
    with open(save_path, "wb") as f:
        f.write(serialization.to_bytes(state_dict))
    print(f"  Policy saved to {save_path}")


def train_curriculum(
    train_config: TrainConfig,
    base_env_config: EnvConfig,
    seed: int = 42,
    log_dir: str = "runs",
    save_dir: str = "checkpoints",
):
    """Train through all curriculum stages.

    Args:
        train_config: Training hyperparameters
        base_env_config: Base environment configuration
        seed: Random seed
        log_dir: Directory for tensorboard logs
        save_dir: Directory for saving checkpoints

    Returns:
        final_runner_state: RunnerState with trained policies
        all_metrics: Dict of metrics from all stages
    """
    os.makedirs(save_dir, exist_ok=True)

    curriculum = get_default_curriculum()
    all_metrics = {}
    runner_state = None

    for stage_idx, stage in enumerate(curriculum):
        print(f"\n{'=' * 60}", flush=True)
        print(f"Stage {stage_idx + 1}/{len(curriculum)}: {stage.name}", flush=True)
        print(f"  Prey: {stage.n_prey}, World: {stage.world_size}x{stage.world_size}", flush=True)
        print(
            f"  Prey speed: {stage.prey_speed_mult}x, Predator speed: {stage.predator_speed_mult}x",
            flush=True,
        )
        print(f"  Max steps: {stage.max_steps}", flush=True)
        print(f"{'=' * 60}", flush=True)

        # Apply stage config
        env_config = EnvConfig(
            n_predators=1,
            n_prey=stage.n_prey,
            world_size=stage.world_size,
            max_steps=stage.max_steps,
            max_speed=base_env_config.max_speed,
            max_acceleration=base_env_config.max_acceleration,
            dt=base_env_config.dt,
            separation_weight=base_env_config.separation_weight,
            alignment_weight=base_env_config.alignment_weight,
            cohesion_weight=base_env_config.cohesion_weight,
            perception_radius=base_env_config.perception_radius,
            capture_radius=base_env_config.capture_radius,
            predator_speed_bonus=stage.predator_speed_mult,
            k_nearest_same=base_env_config.k_nearest_same,
            k_nearest_enemy=base_env_config.k_nearest_enemy,
            prey_learn=False,
            distance_reward=True,
            prey_speed_mult=stage.prey_speed_mult,
            current_stage=stage_idx,
        )

        # Train this stage
        train_config_stage = train_config._replace(total_timesteps=TIMESTEPS_PER_STAGE)
        log_dir_stage = os.path.join(log_dir, stage.name)

        runner_state, metrics = train(
            train_config_stage,
            env_config,
            seed=seed + stage_idx,
            verbose=True,
            log_dir=log_dir_stage,
        )
        all_metrics[stage.name] = metrics

        # Save policy after stage
        checkpoint_path = os.path.join(save_dir, f"stage_{stage_idx + 1}.ckpt")
        save_predator_policy(runner_state.pred_state, checkpoint_path)

    print(f"\n{'=' * 60}")
    print("Curriculum training complete!")
    print(f"Final policy saved to {os.path.join(save_dir, 'stage_4.ckpt')}")
    print(f"{'=' * 60}")

    return runner_state, all_metrics

    if __name__ == "__main__":
        train_config = TrainConfig(
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            n_steps=128,
            n_epochs=4,
            n_minibatches=4,
            n_envs=32,
        )

        base_env_config = EnvConfig(
            separation_weight=1.5,
            alignment_weight=1.0,
            cohesion_weight=1.0,
            perception_radius=15.0,
            capture_radius=0.5,
            predator_speed_bonus=1.5,
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
            log_dir="runs/curriculum",
            save_dir="checkpoints/curriculum",
        )
