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

    def _env_step(runner_state: RunnerState, _):
        """Single environment step for rollout collection."""
        pred_state, env_states, obs, key, update_step = runner_state
        key, k1, k2, k3 = jax.random.split(key, 4)

        n_pred = env_config.n_predators
        n_prey = env_config.n_prey

        # Predator actions from learned policy
        pred_obs_flat = obs["predator"].reshape(-1, env.observation_size)
        pred_out = jax.vmap(lambda o: pred_state.apply_fn(pred_state.params, o))(pred_obs_flat)
        pred_pi = make_distribution(pred_out.action_mean, pred_out.action_logstd)
        pred_actions_flat = pred_pi.sample(seed=k1)
        pred_log_probs_flat = pred_pi.log_prob(pred_actions_flat)

        # Prey actions: random Brownian motion
        prey_actions_flat = jax.random.normal(k2, (config.n_envs * n_prey, env.action_size))
        prey_actions_flat = prey_actions_flat * config.prey_noise_scale

        # Reshape actions
        pred_actions = pred_actions_flat.reshape(config.n_envs, n_pred, -1)
        prey_actions = prey_actions_flat.reshape(config.n_envs, n_prey, -1)
        pred_log_probs = pred_log_probs_flat.reshape(config.n_envs, n_pred)
        pred_values = pred_out.value.reshape(config.n_envs, n_pred)

        # Environment step
        actions = {"predator": pred_actions, "prey": prey_actions}
        step_keys = jax.random.split(k3, config.n_envs)
        next_obs, env_states_new, rewards, dones, info = jax.vmap(env.step)(
            step_keys, env_states, actions
        )

        # Build transition (predator only)
        transition_pred = Transition(
            obs=obs["predator"],
            action=pred_actions,
            reward=rewards["predator"],
            done=dones["predator"],
            log_prob=pred_log_probs,
            value=pred_values,
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

        runner_state = RunnerState(pred_state, env_states, obs, key, update_step)
        return runner_state, (transition_pred, info)

    def _update_step(runner_state: RunnerState, _):
        """Single training update: rollout + PPO update for predators."""
        runner_state, (transitions_pred, infos) = jax.lax.scan(
            _env_step, runner_state, None, length=config.n_steps
        )

        key = runner_state.key
        k1, key = jax.random.split(key)

        pred_state, pred_metrics = ppo_update(
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

        # Metrics
        metrics = {
            "policy_loss": pred_metrics["policy_loss"],
            "value_loss": pred_metrics["value_loss"],
            "entropy": pred_metrics["entropy"],
            "approx_kl": pred_metrics["approx_kl"],
            "reward": transitions_pred.reward.mean(),
            "prey_alive": infos["prey_alive"].mean(),
        }

        # Log metrics
        if log_fn is not None:
            step = runner_state.update_step * steps_per_update
            jax.debug.callback(log_fn, step, metrics)

        # Save checkpoint periodically
        if checkpoint_fn is not None:
            jax.debug.callback(checkpoint_fn, runner_state.update_step, pred_state.params)

        runner_state = RunnerState(
            pred_state,
            runner_state.env_states,
            runner_state.obs,
            key,
            runner_state.update_step + 1,
        )
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
        total_timesteps=5_000_000,
        n_envs=32,
        n_steps=128,
        prey_noise_scale=0.3,
    )
    env_config = EnvConfig()
    train(train_config, env_config)
