"""Single-agent training: one side learns, the other moves randomly.

By default predators learn and prey move randomly. Pass learner="prey"
to invert: prey learn, predators move randomly.
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

    learner_state: TrainState
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
    opponent_noise_scale: chex.Array
    normalize_returns: chex.Array  # bool as array


def make_train(
    n_steps: int,
    n_envs: int,
    n_epochs: int,
    n_minibatches: int,
    n_updates: int,
    env: PredatorPreyEnv,
    env_config: EnvConfig,
    learner: str = "predator",
    opponent_state: "TrainState | None" = None,
):
    """Create a JIT-compiled training function.

    Only shape-affecting params (n_steps, n_epochs, etc.) go in the closure.
    Scalar hyperparams (lr, clip_eps, etc.) are passed as JAX arrays to
    avoid recompilation when only scalars change.

    Args:
        learner: Which side learns — "predator" or "prey".
        opponent_state: Frozen TrainState for the opponent. If provided,
            opponent uses this learned policy instead of random actions.

    Returns (train_fn, steps_per_update). train_fn signature:
        train_fn(learner_state, env_states, obs, key, hparams) -> (runner_state, metrics)
    """
    opponent = "prey" if learner == "predator" else "predator"
    steps_per_update = n_steps * n_envs

    def _update_step(carry, _):
        """Single training update: rollout + PPO update for learner."""
        runner_state, hparams = carry
        learner_state, env_states, obs, key, update_step = runner_state

        # Opponent uses frozen checkpoint if provided, otherwise random
        if opponent_state is not None:
            opp_policy = PolicyConfig(
                PolicyType.LEARNED,
                train_state=opponent_state,
                noise_scale=0.0,
            )
        else:
            opp_policy = PolicyConfig(PolicyType.RANDOM, noise_scale=hparams.opponent_noise_scale)

        policies = {
            learner: PolicyConfig(PolicyType.LEARNED, train_state=learner_state, noise_scale=0.0),
            opponent: opp_policy,
        }
        rollout_config = RolloutConfig(n_steps=n_steps, n_envs=n_envs)

        key, (transitions, infos), obs, env_states = collect_rollouts(
            env, policies, env_config, rollout_config, key, obs, env_states
        )

        k1, key = jax.random.split(key)
        learner_state, learner_metrics, _, _ = ppo_update(
            learner_state,
            transitions[learner],
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
            "policy_loss": learner_metrics["policy_loss"],
            "value_loss": learner_metrics["value_loss"],
            "entropy": learner_metrics["entropy"],
            "approx_kl": learner_metrics["approx_kl"],
            "reward": transitions[learner].reward.mean(),
            "prey_alive": infos["prey_alive"].mean(),
        }

        runner_state = RunnerState(learner_state, env_states, obs, key, update_step + 1)
        return (runner_state, hparams), metrics

    def train_fn(learner_state, env_states, obs, key, hparams):
        """Full training loop — no CPU-GPU sync. Caller should jit (or vmap+jit)."""
        runner_state = RunnerState(learner_state, env_states, obs, key, 0)
        (runner_state, _), metrics = jax.lax.scan(
            _update_step, (runner_state, hparams), None, length=n_updates
        )
        return runner_state, metrics

    return train_fn, steps_per_update


# Module-level cache: same (n_steps, n_envs, n_epochs, n_minibatches, n_updates)
# returns the same jitted function, so JAX reuses the compiled XLA program.
_train_fn_cache: dict = {}


def get_train_fn(
    config: TrainConfig,
    env: PredatorPreyEnv,
    env_config: EnvConfig,
    learner: str = "predator",
    opponent_state: "TrainState | None" = None,
):
    """Get a (cached) JIT-compiled training function.

    Configs with the same shape-affecting params share one compilation.
    Cache is bypassed when opponent_state is provided (frozen params
    are baked into the compiled function).
    """
    n_updates = config.total_timesteps // (config.n_steps * config.n_envs)

    # Frozen opponent params are closed over — can't reuse cached function
    if opponent_state is not None:
        train_fn, steps_per_update = make_train(
            config.n_steps,
            config.n_envs,
            config.n_epochs,
            config.n_minibatches,
            n_updates,
            env,
            env_config,
            learner=learner,
            opponent_state=opponent_state,
        )
        return jax.jit(train_fn), steps_per_update

    cache_key = (
        config.n_steps,
        config.n_envs,
        config.n_epochs,
        config.n_minibatches,
        n_updates,
        learner,
    )
    if cache_key not in _train_fn_cache:
        train_fn, steps_per_update = make_train(
            config.n_steps,
            config.n_envs,
            config.n_epochs,
            config.n_minibatches,
            n_updates,
            env,
            env_config,
            learner=learner,
        )
        _train_fn_cache[cache_key] = (jax.jit(train_fn), steps_per_update)

    return _train_fn_cache[cache_key]


def train(
    config: TrainConfig,
    env_config: EnvConfig,
    seed: int = 0,
    verbose: bool = True,
    log_dir: str | None = "runs",
    learner: str = "predator",
    opponent_checkpoint: str | None = None,
):
    """Train one side while the other moves randomly or uses a frozen checkpoint.

    Args:
        config: Training hyperparameters
        env_config: Environment configuration
        seed: Random seed
        verbose: Whether to print progress
        log_dir: Directory for tensorboard logs. None to disable.
        learner: Which side learns — "predator" or "prey".
        opponent_checkpoint: Path to checkpoint directory for frozen opponent.
            If provided, opponent uses this trained policy instead of random.

    Returns:
        final_state: RunnerState with trained network
        metrics: Training metrics over time
    """
    import pathlib
    from datetime import datetime

    opponent = "prey" if learner == "predator" else "predator"
    prefix = "pred" if learner == "predator" else "prey"
    opp_mode = "frozen" if opponent_checkpoint else "random"

    env = PredatorPreyEnv(env_config)
    steps_per_update = config.n_steps * config.n_envs
    n_updates = config.total_timesteps // steps_per_update

    writer = None
    run_dir = None

    if log_dir is not None:
        from tensorboardX import SummaryWriter

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{learner}_vs_{opp_mode}_{timestamp}"
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
                    "learner": learner,
                    "opponent_checkpoint": opponent_checkpoint,
                },
                f,
                indent=2,
            )

    if verbose:
        print(f"Single-agent training: {learner} learns, {opponent} {opp_mode}")
        if opponent_checkpoint:
            print(f"  Opponent checkpoint: {opponent_checkpoint}")
        print(f"Training for {config.total_timesteps:,} steps ({n_updates} updates)")
        if writer is not None:
            print(f"Logging to {writer.logdir}")
        print("Compiling training function...")
    else:
        print(f"Starting training: {config.total_timesteps:,} steps ({n_updates} updates)")

    # Init outside JIT so orthogonal_init/lr_anneal don't trigger recompilation
    key = jax.random.PRNGKey(seed)
    k1, k2, key = jax.random.split(key, 3)

    learner_state = create_train_state(
        k1,
        env.observation_size,
        env.action_size,
        config.lr,
        config.max_grad_norm,
        total_updates=n_updates if config.lr_anneal else None,
        orthogonal_init=config.orthogonal_init,
        min_lr=config.min_lr,
    )

    # Load frozen opponent if checkpoint provided
    opp_state = None
    if opponent_checkpoint:
        k3, key = jax.random.split(key)
        # Create a dummy TrainState with matching architecture, then swap in loaded params
        opp_state = create_train_state(
            k3,
            env.observation_size,
            env.action_size,
            lr=1e-4,
            max_grad_norm=0.5,
        )
        checkpointer = ocp.PyTreeCheckpointer()
        ckpt_path = str(pathlib.Path(opponent_checkpoint).resolve())
        loaded_params = checkpointer.restore(ckpt_path)
        opp_state = opp_state.replace(params=loaded_params)
        if verbose:
            print(f"Loaded frozen {opponent} checkpoint")

    env_keys = jax.random.split(k2, config.n_envs)
    obs, env_states = jax.vmap(env.reset)(env_keys)

    # Scalar hparams as JAX arrays — changing these does NOT trigger recompilation
    hparams = ScalarHParams(
        gamma=jnp.float32(config.gamma),
        gae_lambda=jnp.float32(config.gae_lambda),
        clip_eps=jnp.float32(config.clip_eps),
        vf_coef=jnp.float32(config.vf_coef),
        ent_coef=jnp.float32(config.ent_coef),
        opponent_noise_scale=jnp.float32(config.prey_noise_scale),
        normalize_returns=jnp.bool_(config.normalize_returns),
    )

    # Cached JIT — only compiles once per unique shape combo
    train_fn, _ = get_train_fn(
        config,
        env,
        env_config,
        learner=learner,
        opponent_state=opp_state,
    )
    runner_state, metrics = train_fn(learner_state, env_states, obs, key, hparams)

    # Batch-write all logs after training (single CPU sync point)
    if writer is not None:
        for i in range(n_updates):
            step = i * steps_per_update
            writer.add_scalar(f"{prefix}/policy_loss", float(metrics["policy_loss"][i]), step)
            writer.add_scalar(f"{prefix}/value_loss", float(metrics["value_loss"][i]), step)
            writer.add_scalar(f"{prefix}/entropy", float(metrics["entropy"][i]), step)
            writer.add_scalar(f"{prefix}/approx_kl", float(metrics["approx_kl"][i]), step)
            writer.add_scalar(f"{prefix}/reward", float(metrics["reward"][i]), step)
            writer.add_scalar("env/prey_alive", float(metrics["prey_alive"][i]), step)
        writer.flush()

    # Save final checkpoint
    if run_dir is not None:
        checkpointer = ocp.PyTreeCheckpointer()
        checkpoint_dir = (run_dir / "checkpoint").resolve()
        checkpointer.save(str(checkpoint_dir), runner_state.learner_state.params, force=True)
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
