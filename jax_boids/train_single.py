"""Single-agent training: predators learn, prey move randomly.

Simplified version to verify the PPO implementation works correctly.
Prey use Brownian motion (random velocity perturbations).
"""

import json
from typing import Dict, NamedTuple

import chex
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState

from jax_boids.collector import PolicyConfig, PolicyType, RolloutConfig, collect_rollouts
from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.types import BoidsState, EnvConfig, TrainConfig
from jax_boids.ppo import (
    create_train_state,
    ppo_update,
)


class RunnerState(NamedTuple):
    """State carried through the training loop."""

    pred_state: TrainState
    env_states: BoidsState
    obs: Dict[str, chex.Array]
    key: chex.PRNGKey
    update_step: int


class ScalarHParams(NamedTuple):
    """Scalar hyperparameters passed as JAX arrays to avoid recompilation."""

    gamma: chex.Array
    gae_lambda: chex.Array
    clip_eps: chex.Array
    vf_coef: chex.Array
    ent_coef: chex.Array
    prey_noise_scale: chex.Array
    normalize_returns: chex.Array  # bool as array


def make_train(
    n_steps: int,
    n_envs: int,
    n_epochs: int,
    n_minibatches: int,
    n_updates: int,
    env: PredatorPreyEnv,
    env_config: EnvConfig,
):
    """Create a JIT-compiled training function.

    Only shape-affecting params (n_steps, n_epochs, etc.) go in the closure.
    Scalar hyperparams (lr, clip_eps, etc.) are passed as JAX arrays to
    avoid recompilation when only scalars change.

    Returns (train_fn, steps_per_update). train_fn signature:
        train_fn(pred_state, env_states, obs, key, hparams) -> (runner_state, metrics)
    """
    steps_per_update = n_steps * n_envs

    def _update_step(carry, _):
        """Single training update: rollout + PPO update for predators."""
        runner_state, hparams = carry
        pred_state, env_states, obs, key, update_step = runner_state

        policies = {
            "predator": PolicyConfig(PolicyType.LEARNED, train_state=pred_state, noise_scale=0.0),
            "prey": PolicyConfig(PolicyType.RANDOM, noise_scale=hparams.prey_noise_scale),
        }
        rollout_config = RolloutConfig(n_steps=n_steps, n_envs=n_envs)

        key, (transitions, infos), obs, env_states = collect_rollouts(
            env, policies, env_config, rollout_config, key, obs, env_states
        )

        k1, key = jax.random.split(key)
        pred_state, pred_metrics, _, _ = ppo_update(
            pred_state,
            transitions["predator"],
            k1,
            hparams.gamma,
            hparams.gae_lambda,
            hparams.clip_eps,
            hparams.vf_coef,
            hparams.ent_coef,
            n_epochs,
            n_minibatches,
            normalize_returns=hparams.normalize_returns,
        )

        metrics = {
            "policy_loss": pred_metrics["policy_loss"],
            "value_loss": pred_metrics["value_loss"],
            "entropy": pred_metrics["entropy"],
            "approx_kl": pred_metrics["approx_kl"],
            "reward": transitions["predator"].reward.mean(),
            "prey_alive": infos["prey_alive"].mean(),
        }

        runner_state = RunnerState(pred_state, env_states, obs, key, update_step + 1)
        return (runner_state, hparams), metrics

    def train_fn(pred_state, env_states, obs, key, hparams):
        """Full training loop — no CPU-GPU sync. Caller should jit (or vmap+jit)."""
        runner_state = RunnerState(pred_state, env_states, obs, key, 0)
        (runner_state, _), metrics = jax.lax.scan(
            _update_step, (runner_state, hparams), None, length=n_updates
        )
        return runner_state, metrics

    return train_fn, steps_per_update


# Module-level cache: same (n_steps, n_envs, n_epochs, n_minibatches, n_updates)
# returns the same jitted function, so JAX reuses the compiled XLA program.
_train_fn_cache: dict = {}


def get_train_fn(config: TrainConfig, env: PredatorPreyEnv, env_config: EnvConfig):
    """Get a cached JIT-compiled training function.

    Configs with the same shape-affecting params (n_steps, n_envs, n_epochs,
    n_minibatches, n_updates) share one compilation. Only the first call per
    unique combo pays the compilation cost.
    """
    n_updates = config.total_timesteps // (config.n_steps * config.n_envs)
    cache_key = (config.n_steps, config.n_envs, config.n_epochs, config.n_minibatches, n_updates)

    if cache_key not in _train_fn_cache:
        train_fn, steps_per_update = make_train(
            config.n_steps,
            config.n_envs,
            config.n_epochs,
            config.n_minibatches,
            n_updates,
            env,
            env_config,
        )
        _train_fn_cache[cache_key] = (jax.jit(train_fn), steps_per_update)

    return _train_fn_cache[cache_key]


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
    steps_per_update = config.n_steps * config.n_envs
    n_updates = config.total_timesteps // steps_per_update

    writer = None
    run_dir = None

    if log_dir is not None:
        from tensorboardX import SummaryWriter

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"pred_vs_random_{timestamp}"
        run_dir = pathlib.Path(log_dir) / run_name
        writer = SummaryWriter(log_dir=str(run_dir))
        writer.add_text("config/train", str(config._asdict()))
        writer.add_text("config/env", str(env_config.__dict__))

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

    if verbose:
        noise = config.prey_noise_scale
        print(f"Single-agent training: predators learn, prey random (noise={noise})")
        print(f"Training for {config.total_timesteps:,} steps ({n_updates} updates)")
        if writer is not None:
            print(f"Logging to {writer.logdir}")
        print("Compiling training function...")
    else:
        print(f"Starting training: {config.total_timesteps:,} steps ({n_updates} updates)")

    # Init outside JIT so orthogonal_init/lr_anneal don't trigger recompilation
    key = jax.random.PRNGKey(seed)
    k1, k2, key = jax.random.split(key, 3)

    pred_state = create_train_state(
        k1,
        env.observation_size,
        env.action_size,
        config.lr,
        config.max_grad_norm,
        total_updates=n_updates if config.lr_anneal else None,
        orthogonal_init=config.orthogonal_init,
        min_lr=config.min_lr,
    )

    env_keys = jax.random.split(k2, config.n_envs)
    obs, env_states = jax.vmap(env.reset)(env_keys)

    # Scalar hparams as JAX arrays — changing these does NOT trigger recompilation
    hparams = ScalarHParams(
        gamma=jnp.float32(config.gamma),
        gae_lambda=jnp.float32(config.gae_lambda),
        clip_eps=jnp.float32(config.clip_eps),
        vf_coef=jnp.float32(config.vf_coef),
        ent_coef=jnp.float32(config.ent_coef),
        prey_noise_scale=jnp.float32(config.prey_noise_scale),
        normalize_returns=jnp.bool_(config.normalize_returns),
    )

    # Cached JIT — only compiles once per unique (n_steps, n_envs, n_epochs) combo
    train_fn, _ = get_train_fn(config, env, env_config)
    runner_state, metrics = train_fn(pred_state, env_states, obs, key, hparams)

    # Batch-write all logs after training (single CPU sync point)
    if writer is not None:
        for i in range(n_updates):
            step = i * steps_per_update
            writer.add_scalar("pred/policy_loss", float(metrics["policy_loss"][i]), step)
            writer.add_scalar("pred/value_loss", float(metrics["value_loss"][i]), step)
            writer.add_scalar("pred/entropy", float(metrics["entropy"][i]), step)
            writer.add_scalar("pred/approx_kl", float(metrics["approx_kl"][i]), step)
            writer.add_scalar("pred/reward", float(metrics["reward"][i]), step)
            writer.add_scalar("env/prey_alive", float(metrics["prey_alive"][i]), step)
        writer.flush()

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
