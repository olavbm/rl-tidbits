"""IPPO training for predator-prey boids.

Uses lax.scan for fast JIT compilation (purejaxrl pattern).
Both predators and prey learn via PPO.
"""

from typing import Dict, NamedTuple

import chex
import jax
from flax.training.train_state import TrainState

from jax_boids.envs.curriculum import advance_curriculum
from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.types import BoidsState, EnvConfig
from jax_boids.ppo import (
    Transition,
    create_train_state,
    make_distribution,
    ppo_update,
    select_on_reset,
)


class TrainConfig(NamedTuple):
    """Training hyperparameters for multi-agent training."""

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


class RunnerState(NamedTuple):
    """State carried through the training loop."""

    pred_state: TrainState
    prey_state: TrainState
    env_states: BoidsState
    obs: Dict[str, chex.Array]
    key: chex.PRNGKey
    update_step: int
    env_config: EnvConfig


def make_train(config: TrainConfig, env: PredatorPreyEnv, env_config: EnvConfig, log_fn=None):
    """Create the JIT-compiled training function.

    Both predators and prey learn via PPO.

    Args:
        log_fn: Optional callback for logging metrics. Called as log_fn(step, metrics)
    """
    steps_per_update = config.n_steps * config.n_envs
    n_pred = env_config.n_predators
    n_prey = env_config.n_prey

    def _env_step(runner_state: RunnerState, _):
        """Single environment step for rollout collection."""
        pred_state, prey_state, env_states, obs, key, update_step, _ = runner_state
        key, k1, k2, k3 = jax.random.split(key, 4)

        # Flatten obs: [n_envs, n_agents, obs_size] -> [n_envs * n_agents, obs_size]
        pred_obs_flat = obs["predator"].reshape(-1, env.observation_size)
        prey_obs_flat = obs["prey"].reshape(-1, env.observation_size)

        # Get network outputs
        pred_out = jax.vmap(lambda o: pred_state.apply_fn(pred_state.params, o))(pred_obs_flat)
        prey_out = jax.vmap(lambda o: prey_state.apply_fn(prey_state.params, o))(prey_obs_flat)

        # Sample actions
        pred_pi = make_distribution(pred_out.action_mean, pred_out.action_logstd)
        prey_pi = make_distribution(prey_out.action_mean, prey_out.action_logstd)

        pred_actions_flat = pred_pi.sample(seed=k1)
        prey_actions_flat = prey_pi.sample(seed=k2)
        pred_log_probs_flat = pred_pi.log_prob(pred_actions_flat)
        prey_log_probs_flat = prey_pi.log_prob(prey_actions_flat)

        # Reshape back to [n_envs, n_agents, ...]
        pred_actions = pred_actions_flat.reshape(config.n_envs, n_pred, -1)
        prey_actions = prey_actions_flat.reshape(config.n_envs, n_prey, -1)
        pred_log_probs = pred_log_probs_flat.reshape(config.n_envs, n_pred)
        prey_log_probs = prey_log_probs_flat.reshape(config.n_envs, n_prey)
        pred_values = pred_out.value.reshape(config.n_envs, n_pred)
        prey_values = prey_out.value.reshape(config.n_envs, n_prey)

        # Environment step
        actions = {"predator": pred_actions, "prey": prey_actions}
        step_keys = jax.random.split(k3, config.n_envs)
        next_obs, env_states_new, rewards, dones, info = jax.vmap(env.step)(
            step_keys, env_states, actions
        )

        # Build transitions
        transition_pred = Transition(
            obs=obs["predator"],
            action=pred_actions,
            reward=rewards["predator"],
            done=dones["predator"],
            log_prob=pred_log_probs,
            value=pred_values,
        )
        transition_prey = Transition(
            obs=obs["prey"],
            action=prey_actions,
            reward=rewards["prey"],
            done=dones["prey"],
            log_prob=prey_log_probs,
            value=prey_values,
        )

        # Handle resets
        reset_mask = dones["__all__"]
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, config.n_envs)
        new_obs, new_states = jax.vmap(env.reset)(reset_keys)

        def _select(new, old):
            return select_on_reset(reset_mask, new, old)

        obs = jax.tree.map(_select, new_obs, next_obs)
        env_states = jax.tree.map(_select, new_states, env_states_new)

        runner_state = RunnerState(
            pred_state, prey_state, env_states, obs, key, update_step, runner_state.env_config
        )
        return runner_state, (transition_pred, transition_prey, info)

    def _update_step(runner_state: RunnerState, _):
        """Single training update: rollout + PPO update."""
        # Collect rollout using lax.scan
        runner_state, (transitions_pred, transitions_prey, infos) = jax.lax.scan(
            _env_step, runner_state, None, length=config.n_steps
        )

        # PPO updates for both teams
        key = runner_state.key
        k1, k2, key = jax.random.split(key, 3)

        pred_state, pred_metrics, pred_adv_mean, pred_adv_std = ppo_update(
            runner_state.pred_state,
            transitions_pred,
            k1,
            config.gamma,
            config.gae_lambda,
            config.clip_eps,
            config.vf_coef,
            config.ent_coef,
            config.n_epochs,
            config.n_minibatches,
        )
        prey_state, prey_metrics, prey_adv_mean, prey_adv_std = ppo_update(
            runner_state.prey_state,
            transitions_prey,
            k2,
            config.gamma,
            config.gae_lambda,
            config.clip_eps,
            config.vf_coef,
            config.ent_coef,
            config.n_epochs,
            config.n_minibatches,
        )

        # Combine metrics
        metrics = {"pred_" + k: v for k, v in pred_metrics.items()}
        metrics.update({"prey_" + k: v for k, v in prey_metrics.items()})
        metrics["pred_advantages_mean"] = pred_adv_mean
        metrics["pred_advantages_std"] = pred_adv_std
        metrics["prey_advantages_mean"] = prey_adv_mean
        metrics["prey_advantages_std"] = prey_adv_std
        metrics["prey_alive"] = infos["prey_alive"].mean()
        metrics["pred_reward"] = transitions_pred.reward.mean()
        metrics["prey_reward"] = transitions_prey.reward.mean()
        metrics["curriculum_stage"] = runner_state.env_config.current_stage

        # Log metrics using jax.debug.callback (JAX-native logging)
        if log_fn is not None:
            step = runner_state.update_step * steps_per_update
            jax.debug.callback(log_fn, step, metrics)

        runner_state = RunnerState(
            pred_state,
            prey_state,
            runner_state.env_states,
            runner_state.obs,
            key,
            runner_state.update_step + 1,
            runner_state.env_config,
        )
        return runner_state, metrics

    def train_fn(key: chex.PRNGKey):
        """Full training loop - JIT compiled with logging via debug.callback."""
        k1, k2, k3, key = jax.random.split(key, 4)

        pred_state = create_train_state(
            k1, env.observation_size, env.action_size, config.lr, config.max_grad_norm
        )
        prey_state = create_train_state(
            k2, env.observation_size, env.action_size, config.lr, config.max_grad_norm
        )

        env_keys = jax.random.split(k3, config.n_envs)
        obs, env_states = jax.vmap(env.reset)(env_keys)

        runner_state = RunnerState(pred_state, prey_state, env_states, obs, key, 0, env_config)

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
    """Train predator-prey agents.

    Uses jax.debug.callback for live logging during JIT-compiled training.

    Args:
        config: Training hyperparameters
        env_config: Environment configuration
        seed: Random seed
        verbose: Whether to print progress
        log_dir: Directory for tensorboard logs. None to disable logging.

    Returns:
        final_state: RunnerState with trained networks
        metrics: Training metrics over time
    """
    from datetime import datetime

    env = PredatorPreyEnv(env_config)
    n_updates = config.total_timesteps // (config.n_steps * config.n_envs)

    # Setup tensorboard logging callback
    writer = None
    log_fn = None

    if log_dir is not None:
        from tensorboardX import SummaryWriter

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"predprey_{timestamp}"
        writer = SummaryWriter(log_dir=f"{log_dir}/{run_name}")
        writer.add_text("config/train", str(config._asdict()))
        writer.add_text("config/env", str(env_config.__dict__))

        def log_fn(step, metrics):
            """Logging callback called from inside JIT via jax.debug.callback."""
            step = int(step)
            writer.add_scalar("pred/policy_loss", float(metrics["pred_policy_loss"]), step)
            writer.add_scalar("pred/value_loss", float(metrics["pred_value_loss"]), step)
            writer.add_scalar("pred/entropy", float(metrics["pred_entropy"]), step)
            writer.add_scalar("pred/approx_kl", float(metrics["pred_approx_kl"]), step)
            writer.add_scalar("pred/reward", float(metrics["pred_reward"]), step)
            writer.add_scalar("pred/advantages_mean", float(metrics["pred_advantages_mean"]), step)
            writer.add_scalar("pred/advantages_std", float(metrics["pred_advantages_std"]), step)
            writer.add_scalar("prey/policy_loss", float(metrics["prey_policy_loss"]), step)
            writer.add_scalar("prey/value_loss", float(metrics["prey_value_loss"]), step)
            writer.add_scalar("prey/entropy", float(metrics["prey_entropy"]), step)
            writer.add_scalar("prey/approx_kl", float(metrics["prey_approx_kl"]), step)
            writer.add_scalar("prey/reward", float(metrics["prey_reward"]), step)
            writer.add_scalar("prey/advantages_mean", float(metrics["prey_advantages_mean"]), step)
            writer.add_scalar("prey/advantages_std", float(metrics["prey_advantages_std"]), step)
            writer.add_scalar("env/prey_alive", float(metrics["prey_alive"]), step)
            writer.add_scalar("curriculum/stage", float(metrics["curriculum_stage"]), step)
            writer.flush()
            if verbose:
                print(f"  Step {step:,} - prey_alive: {float(metrics['prey_alive']):.1f}")

    if verbose:
        print(f"Training for {config.total_timesteps:,} steps ({n_updates} updates)")
        if writer is not None:
            print(f"Logging to {writer.logdir}")
        print("Compiling training function (this takes a moment)...")

    train_fn = make_train(config, env, env_config, log_fn=log_fn)
    train_fn = jax.jit(train_fn)

    key = jax.random.PRNGKey(seed)
    runner_state, metrics = train_fn(key)

    if writer is not None:
        writer.close()

    if verbose:
        print("Training complete!")
        print(f"  Final pred policy_loss: {metrics['pred_policy_loss'][-1]:.4f}")
        print(f"  Final prey policy_loss: {metrics['prey_policy_loss'][-1]:.4f}")
        print(f"  Final prey_alive: {metrics['prey_alive'][-1]:.1f}")

    return runner_state, metrics


if __name__ == "__main__":
    train_config = TrainConfig(
        total_timesteps=20_000_000,
        n_envs=32,
        n_steps=128,
    )
    env_config = EnvConfig()
    train(train_config, env_config)
