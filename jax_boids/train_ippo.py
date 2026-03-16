"""IPPO training for predator-prey boids.

Both predators and prey learn via independent PPO with separate hyperparameters.
Uses collector.py for rollouts, ScalarHParams for JIT reuse,
and batch logging (no CPU-GPU sync during training).
"""

import json
from typing import NamedTuple

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
    """State carried through the IPPO training loop."""

    pred_state: TrainState
    prey_state: TrainState
    env_states: BoidsState
    obs: dict[str, chex.Array]
    key: chex.PRNGKey
    update_step: int


class ScalarHParams(NamedTuple):
    """Scalar hyperparameters for one agent side, passed as JAX arrays."""

    gamma: chex.Array
    gae_lambda: chex.Array
    clip_eps: chex.Array
    vf_coef: chex.Array
    ent_coef: chex.Array
    normalize_returns: chex.Array  # bool as array


def _resolve(prey_val, pred_val):
    """Resolve prey override: use prey value if set, else predator default."""
    return prey_val if prey_val is not None else pred_val


def resolve_prey_hparams(config: TrainConfig) -> dict:
    """Resolve prey hyperparameters, falling back to predator values.

    Returns dict with resolved prey lr, gamma, gae_lambda, etc.
    """
    return {
        "lr": _resolve(config.prey_lr, config.lr),
        "gamma": _resolve(config.prey_gamma, config.gamma),
        "gae_lambda": _resolve(config.prey_gae_lambda, config.gae_lambda),
        "clip_eps": _resolve(config.prey_clip_eps, config.clip_eps),
        "vf_coef": _resolve(config.prey_vf_coef, config.vf_coef),
        "ent_coef": _resolve(config.prey_ent_coef, config.ent_coef),
        "max_grad_norm": _resolve(config.prey_max_grad_norm, config.max_grad_norm),
        "orthogonal_init": _resolve(config.prey_orthogonal_init, config.orthogonal_init),
        "lr_anneal": _resolve(config.prey_lr_anneal, config.lr_anneal),
        "min_lr": _resolve(config.prey_min_lr, config.min_lr),
        "normalize_returns": _resolve(config.prey_normalize_returns, config.normalize_returns),
    }


def make_train(
    n_steps: int,
    n_envs: int,
    n_epochs: int,
    n_minibatches: int,
    n_updates: int,
    env: PredatorPreyEnv,
    env_config: EnvConfig,
):
    """Create a JIT-compiled IPPO training function.

    Only shape-affecting params (n_steps, n_epochs, etc.) go in the closure.
    Scalar hyperparams are passed as JAX arrays to avoid recompilation.

    Returns (train_fn, steps_per_update). train_fn signature:
        train_fn(pred_state, prey_state, env_states, obs, key, pred_hparams, prey_hparams)
            -> (runner_state, metrics)
    """
    steps_per_update = n_steps * n_envs

    def _update_step(carry, _):
        """Single training update: rollout + PPO update for both sides."""
        runner_state, pred_hp, prey_hp = carry
        pred_state, prey_state, env_states, obs, key, update_step = runner_state

        # Both sides use learned policies
        policies = {
            "predator": PolicyConfig(PolicyType.LEARNED, train_state=pred_state, noise_scale=0.0),
            "prey": PolicyConfig(PolicyType.LEARNED, train_state=prey_state, noise_scale=0.0),
        }
        rollout_config = RolloutConfig(n_steps=n_steps, n_envs=n_envs)

        key, (transitions, infos), obs, env_states = collect_rollouts(
            env, policies, env_config, rollout_config, key, obs, env_states
        )

        # PPO update for predators (predator hparams)
        k1, k2, key = jax.random.split(key, 3)
        pred_state, pred_metrics, pred_adv_mean, pred_adv_std = ppo_update(
            pred_state,
            transitions["predator"],
            k1,
            pred_hp.gamma,
            pred_hp.gae_lambda,
            pred_hp.clip_eps,
            pred_hp.vf_coef,
            pred_hp.ent_coef,
            n_epochs,
            n_minibatches,
            normalize_returns=pred_hp.normalize_returns,
        )

        # PPO update for prey (prey hparams)
        prey_state, prey_metrics, prey_adv_mean, prey_adv_std = ppo_update(
            prey_state,
            transitions["prey"],
            k2,
            prey_hp.gamma,
            prey_hp.gae_lambda,
            prey_hp.clip_eps,
            prey_hp.vf_coef,
            prey_hp.ent_coef,
            n_epochs,
            n_minibatches,
            normalize_returns=prey_hp.normalize_returns,
        )

        metrics = {
            "pred_policy_loss": pred_metrics["policy_loss"],
            "pred_value_loss": pred_metrics["value_loss"],
            "pred_entropy": pred_metrics["entropy"],
            "pred_approx_kl": pred_metrics["approx_kl"],
            "pred_adv_mean": pred_adv_mean,
            "pred_adv_std": pred_adv_std,
            "prey_policy_loss": prey_metrics["policy_loss"],
            "prey_value_loss": prey_metrics["value_loss"],
            "prey_entropy": prey_metrics["entropy"],
            "prey_approx_kl": prey_metrics["approx_kl"],
            "prey_adv_mean": prey_adv_mean,
            "prey_adv_std": prey_adv_std,
            "pred_reward": transitions["predator"].reward.mean(),
            "prey_reward": transitions["prey"].reward.mean(),
            "prey_alive": infos["prey_alive"].mean(),
        }

        runner_state = RunnerState(pred_state, prey_state, env_states, obs, key, update_step + 1)
        return (runner_state, pred_hp, prey_hp), metrics

    def train_fn(pred_state, prey_state, env_states, obs, key, pred_hparams, prey_hparams):
        """Full IPPO training loop — no CPU-GPU sync. Caller should jit."""
        runner_state = RunnerState(pred_state, prey_state, env_states, obs, key, 0)
        (runner_state, _, _), metrics = jax.lax.scan(
            _update_step, (runner_state, pred_hparams, prey_hparams), None, length=n_updates
        )
        return runner_state, metrics

    return train_fn, steps_per_update


# Module-level cache: same (n_steps, n_envs, n_epochs, n_minibatches, n_updates)
# returns the same jitted function, so JAX reuses the compiled XLA program.
_train_fn_cache: dict = {}


def get_train_fn(config: TrainConfig, env: PredatorPreyEnv, env_config: EnvConfig):
    """Get a cached JIT-compiled IPPO training function.

    Configs with the same shape-affecting params share one compilation.
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
    """Train both predators and prey via IPPO with separate hyperparameters.

    Args:
        config: Training hyperparameters. Predator values are used as defaults;
            prey_* fields override for the prey side.
        env_config: Environment configuration (must have prey_learn=True)
        seed: Random seed
        verbose: Whether to print progress
        log_dir: Directory for tensorboard logs. None to disable.

    Returns:
        final_state: RunnerState with trained predator and prey networks
        metrics: Training metrics over time
    """
    import pathlib
    from datetime import datetime

    env = PredatorPreyEnv(env_config)
    steps_per_update = config.n_steps * config.n_envs
    n_updates = config.total_timesteps // steps_per_update

    # Resolve prey hyperparameters
    prey_hp = resolve_prey_hparams(config)

    writer = None
    run_dir = None

    if log_dir is not None:
        from tensorboardX import SummaryWriter

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ippo_{timestamp}"
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
                    "prey_hparams": prey_hp,
                    "mode": "ippo",
                },
                f,
                indent=2,
            )

    if verbose:
        print("IPPO training: both predators and prey learn")
        print(f"  {env_config.n_predators} predators, {env_config.n_prey} prey")
        print(f"  pred lr={config.lr:.2e}, prey lr={prey_hp['lr']:.2e}")
        print(f"  pred ent={config.ent_coef:.3f}, prey ent={prey_hp['ent_coef']:.3f}")
        print(f"Training for {config.total_timesteps:,} steps ({n_updates} updates)")
        if writer is not None:
            print(f"Logging to {writer.logdir}")
        print("Compiling training function...")
    else:
        print(f"IPPO training: {config.total_timesteps:,} steps ({n_updates} updates)")

    # Init outside JIT so orthogonal_init/lr_anneal don't trigger recompilation
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, key = jax.random.split(key, 4)

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
    prey_state = create_train_state(
        k2,
        env.observation_size,
        env.action_size,
        prey_hp["lr"],
        prey_hp["max_grad_norm"],
        total_updates=n_updates if prey_hp["lr_anneal"] else None,
        orthogonal_init=prey_hp["orthogonal_init"],
        min_lr=prey_hp["min_lr"],
    )

    env_keys = jax.random.split(k3, config.n_envs)
    obs, env_states = jax.vmap(env.reset)(env_keys)

    # Scalar hparams as JAX arrays — changing these does NOT trigger recompilation
    pred_hparams = ScalarHParams(
        gamma=jnp.float32(config.gamma),
        gae_lambda=jnp.float32(config.gae_lambda),
        clip_eps=jnp.float32(config.clip_eps),
        vf_coef=jnp.float32(config.vf_coef),
        ent_coef=jnp.float32(config.ent_coef),
        normalize_returns=jnp.bool_(config.normalize_returns),
    )
    prey_hparams = ScalarHParams(
        gamma=jnp.float32(prey_hp["gamma"]),
        gae_lambda=jnp.float32(prey_hp["gae_lambda"]),
        clip_eps=jnp.float32(prey_hp["clip_eps"]),
        vf_coef=jnp.float32(prey_hp["vf_coef"]),
        ent_coef=jnp.float32(prey_hp["ent_coef"]),
        normalize_returns=jnp.bool_(prey_hp["normalize_returns"]),
    )

    # Cached JIT — only compiles once per unique (n_steps, n_envs, n_epochs) combo
    train_fn, _ = get_train_fn(config, env, env_config)
    runner_state, metrics = train_fn(
        pred_state, prey_state, env_states, obs, key, pred_hparams, prey_hparams
    )

    # Batch-write all logs after training (single CPU sync point)
    if writer is not None:
        for i in range(n_updates):
            step = i * steps_per_update
            writer.add_scalar("pred/policy_loss", float(metrics["pred_policy_loss"][i]), step)
            writer.add_scalar("pred/value_loss", float(metrics["pred_value_loss"][i]), step)
            writer.add_scalar("pred/entropy", float(metrics["pred_entropy"][i]), step)
            writer.add_scalar("pred/approx_kl", float(metrics["pred_approx_kl"][i]), step)
            writer.add_scalar("pred/reward", float(metrics["pred_reward"][i]), step)
            writer.add_scalar("pred/adv_mean", float(metrics["pred_adv_mean"][i]), step)
            writer.add_scalar("pred/adv_std", float(metrics["pred_adv_std"][i]), step)
            writer.add_scalar("prey/policy_loss", float(metrics["prey_policy_loss"][i]), step)
            writer.add_scalar("prey/value_loss", float(metrics["prey_value_loss"][i]), step)
            writer.add_scalar("prey/entropy", float(metrics["prey_entropy"][i]), step)
            writer.add_scalar("prey/approx_kl", float(metrics["prey_approx_kl"][i]), step)
            writer.add_scalar("prey/reward", float(metrics["prey_reward"][i]), step)
            writer.add_scalar("prey/adv_mean", float(metrics["prey_adv_mean"][i]), step)
            writer.add_scalar("prey/adv_std", float(metrics["prey_adv_std"][i]), step)
            writer.add_scalar("env/prey_alive", float(metrics["prey_alive"][i]), step)
        writer.flush()

    # Save checkpoints for both networks
    if run_dir is not None:
        checkpointer = ocp.PyTreeCheckpointer()
        pred_ckpt_dir = (run_dir / "checkpoint_pred").resolve()
        prey_ckpt_dir = (run_dir / "checkpoint_prey").resolve()
        checkpointer.save(str(pred_ckpt_dir), runner_state.pred_state.params, force=True)
        checkpointer.save(str(prey_ckpt_dir), runner_state.prey_state.params, force=True)
        if verbose:
            print(f"Saved predator checkpoint to {pred_ckpt_dir}")
            print(f"Saved prey checkpoint to {prey_ckpt_dir}")

    if writer is not None:
        writer.close()

    if verbose:
        print("Training complete!")
        print(f"  Pred policy_loss: {metrics['pred_policy_loss'][-1]:.4f}")
        print(f"  Prey policy_loss: {metrics['prey_policy_loss'][-1]:.4f}")
        print(f"  Final prey_alive: {metrics['prey_alive'][-1]:.1f}")

    return runner_state, metrics


if __name__ == "__main__":
    train_config = TrainConfig(
        total_timesteps=5_000_000,
        n_envs=64,
        n_steps=256,
        prey_lr=3e-4,
    )
    env_config = EnvConfig(
        n_predators=2,
        n_prey=5,
        prey_learn=True,
    )
    train(train_config, env_config)
