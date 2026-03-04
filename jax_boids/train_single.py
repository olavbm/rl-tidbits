"""Single-agent training: predators learn, prey move randomly.

Simplified version to verify the PPO implementation works correctly.
Prey use Brownian motion (random velocity perturbations).
"""

import json
from typing import Dict, NamedTuple

import chex
import jax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState

from jax_boids.collector import PolicyConfig, PolicyType, RolloutConfig, collect_rollouts
from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.types import BoidsState, EnvConfig
from jax_boids.ppo import (
    create_train_state,
    ppo_update,
)


class TrainConfig(NamedTuple):
    """Training hyperparameters for single-agent training."""

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_steps: int = 128
    n_epochs: int = 4
    n_minibatches: int = 4
    total_timesteps: int = 1_000_000
    n_envs: int = 32
    prey_noise_scale: float = 0.3  # scale of random prey actions


class RunnerState(NamedTuple):
    """State carried through the training loop."""

    pred_state: TrainState
    env_states: BoidsState
    obs: Dict[str, chex.Array]
    key: chex.PRNGKey
    update_step: int


def make_train(
    config: TrainConfig,
    env: PredatorPreyEnv,
    env_config: EnvConfig,
    log_fn=None,
    checkpoint_fn=None,
):
    """Create the JIT-compiled training function.

    Predators learn via PPO, prey use random actions (Brownian motion).
    """
    steps_per_update = config.n_steps * config.n_envs

    def _update_step(runner_state: RunnerState, _):
        """Single training update: rollout + PPO update for predators."""
        pred_state, env_states, obs, key, update_step = runner_state

        # Configure policies: learned for predators, random for prey
        policies = {
            "predator": PolicyConfig(PolicyType.LEARNED, train_state=pred_state, noise_scale=0.0),
            "prey": PolicyConfig(PolicyType.RANDOM, noise_scale=config.prey_noise_scale),
        }

        # Configure rollout collection
        rollout_config = RolloutConfig(n_steps=config.n_steps, n_envs=config.n_envs)

        # Collect rollouts using the abstraction
        key, (transitions, infos), obs, env_states = collect_rollouts(
            env, policies, env_config, rollout_config, key, obs, env_states
        )

        # Separate key for PPO update
        k1, key = jax.random.split(key)

        # PPO update for predators only
        pred_state, pred_metrics = ppo_update(
            pred_state,
            transitions["predator"],
            k1,
            config.gamma,
            config.gae_lambda,
            config.clip_eps,
            config.vf_coef,
            config.ent_coef,
            config.n_epochs,
            config.n_minibatches,
        )

        # Metrics (keep as JAX arrays for scan compatibility)
        metrics = {
            "policy_loss": pred_metrics["policy_loss"],
            "value_loss": pred_metrics["value_loss"],
            "entropy": pred_metrics["entropy"],
            "approx_kl": pred_metrics["approx_kl"],
            "reward": transitions["predator"].reward.mean(),
            "prey_alive": infos["prey_alive"].mean(),
        }

        # Log metrics
        if log_fn is not None:
            step = update_step * steps_per_update
            jax.debug.callback(log_fn, step, metrics)

        # Save checkpoint periodically
        if checkpoint_fn is not None:
            jax.debug.callback(checkpoint_fn, update_step, pred_state.params)

        runner_state = RunnerState(pred_state, env_states, obs, key, update_step + 1)
        return runner_state, metrics

    def train_fn(key: chex.PRNGKey):
        """Full training loop."""
        k1, k2, key = jax.random.split(key, 3)

        pred_state = create_train_state(k1, env.observation_size, env.action_size, config.lr)

        env_keys = jax.random.split(k2, config.n_envs)
        obs, env_states = jax.vmap(env.reset)(env_keys)

        runner_state = RunnerState(pred_state, env_states, obs, key, 0)

        n_updates = config.total_timesteps // (config.n_steps * config.n_envs)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, length=n_updates)

        return runner_state, metrics

    return train_fn


def train(
    config: TrainConfig,
    env_config: EnvConfig,
    seed: int = 0,
    verbose: bool = True,
    log_dir: str | None = "runs",
):
    """Train predators to catch randomly-moving prey.

    Args:
        config: Training hyperparameters
        env_config: Environment configuration
        seed: Random seed
        verbose: Whether to print progress
        log_dir: Directory for tensorboard logs. None to disable.

    Returns:
        final_state: RunnerState with trained predator network
        metrics: Training metrics over time
    """
    import pathlib
    from datetime import datetime

    env = PredatorPreyEnv(env_config)
    n_updates = config.total_timesteps // (config.n_steps * config.n_envs)

    writer = None
    log_fn = None
    checkpoint_fn = None
    run_dir = None

    if log_dir is not None:
        from tensorboardX import SummaryWriter

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"pred_vs_random_{timestamp}"
        run_dir = pathlib.Path(log_dir) / run_name
        writer = SummaryWriter(log_dir=str(run_dir))
        writer.add_text("config/train", str(config._asdict()))
        writer.add_text("config/env", str(env_config.__dict__))
        writer.add_text("info", "Single-agent: predators learn, prey random")

        # Save configs as JSON for loading later
        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "train": config._asdict(),
                    "env": {k: v for k, v in env_config.__dict__.items() if not k.startswith("_")},
                },
                f,
                indent=2,
            )

        # Setup checkpointer (Orbax requires absolute paths)
        checkpointer = ocp.PyTreeCheckpointer()
        checkpoint_dir = (run_dir / "checkpoint").resolve()

        def checkpoint_fn(update_step, params):
            """Save checkpoint every 100 updates."""
            update_step = int(update_step)
            if update_step > 0 and update_step % 100 == 0:
                checkpointer.save(str(checkpoint_dir), params, force=True)

        def log_fn(step, metrics):
            """Logging callback."""
            step = int(step)
            writer.add_scalar("pred/policy_loss", float(metrics["policy_loss"]), step)
            writer.add_scalar("pred/value_loss", float(metrics["value_loss"]), step)
            writer.add_scalar("pred/entropy", float(metrics["entropy"]), step)
            writer.add_scalar("pred/approx_kl", float(metrics["approx_kl"]), step)
            writer.add_scalar("pred/reward", float(metrics["reward"]), step)
            writer.add_scalar("env/prey_alive", float(metrics["prey_alive"]), step)
            writer.flush()
            if verbose:
                prey = float(metrics["prey_alive"])
                reward = float(metrics["reward"])
                print(f"  Step {step:,} - prey_alive: {prey:.1f}, reward: {reward:.3f}")

    if verbose:
        noise = config.prey_noise_scale
        print(f"Single-agent training: predators learn, prey random (noise={noise})")
        print(f"Training for {config.total_timesteps:,} steps ({n_updates} updates)")
        if writer is not None:
            print(f"Logging to {writer.logdir}")
        print("Compiling training function...")

    train_fn = make_train(config, env, env_config, log_fn=log_fn, checkpoint_fn=checkpoint_fn)
    train_fn = jax.jit(train_fn)

    key = jax.random.PRNGKey(seed)
    runner_state, metrics = train_fn(key)

    # Save final checkpoint
    if run_dir is not None:
        checkpointer = ocp.PyTreeCheckpointer()
        checkpoint_dir = (run_dir / "checkpoint").resolve()
        checkpointer.save(str(checkpoint_dir), runner_state.pred_state.params, force=True)
        if verbose:
            print(f"Saved final checkpoint to {checkpoint_dir}")

    if writer is not None:
        writer.close()

    if verbose:
        print("Training complete!")
        print(f"  Final policy_loss: {metrics['policy_loss'][-1]:.4f}")
        print(f"  Final prey_alive: {metrics['prey_alive'][-1]:.1f}")

    return runner_state, metrics


if __name__ == "__main__":
    train_config = TrainConfig(
        total_timesteps=5_000_000,  # 5M steps for convergence
        n_envs=32,
        n_steps=256,
        prey_noise_scale=0.1,  # Small noise so prey move slightly
    )
    env_config = EnvConfig()
    train(train_config, env_config)
