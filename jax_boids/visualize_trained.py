"""Visualize a trained predator-prey agent."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.types import BoidsState, EnvConfig
from jax_boids.networks import ActorCritic
from jax_boids.ppo import make_distribution
from jax_boids.visualize import animate_episode


def load_checkpoint(run_dir: str):
    """Load params and configs from a training run.

    Auto-detects IPPO vs single-agent based on checkpoint directory structure.
    For IPPO runs that have a timestamped subdirectory, finds it automatically.

    Args:
        run_dir: Path to run directory (e.g., 'runs/train_ippo_5m')

    Returns:
        pred_params: Predator network parameters
        prey_params: Prey network parameters (None for single-agent)
        train_config: Training configuration dict
        env_config: Environment configuration as EnvConfig
    """
    run_path = Path(run_dir).resolve()
    checkpointer = ocp.PyTreeCheckpointer()

    # Find the actual checkpoint directory (may be in a timestamped subdirectory)
    config_path = run_path / "config.json"

    # Check for timestamped subdirectory (e.g. ippo_*, pred_*, prey_*)
    subdirs = [
        d
        for d in run_path.iterdir()
        if d.is_dir() and d.name.startswith(("ippo_", "pred_", "prey_"))
    ]
    if subdirs:
        # Use the most recent subdirectory
        subdir = sorted(subdirs)[-1]
        config_path = subdir / "config.json"
        checkpoint_base = subdir
    else:
        checkpoint_base = run_path

    # Load config
    with open(config_path) as f:
        configs = json.load(f)

    env_config = EnvConfig(**configs["env"])
    is_ippo = configs.get("mode") == "ippo"
    learner = configs.get("learner", "predator")

    # Load checkpoints based on training mode
    if is_ippo:
        pred_params = checkpointer.restore(str(checkpoint_base / "checkpoint_pred"))
        prey_params = checkpointer.restore(str(checkpoint_base / "checkpoint_prey"))
    elif learner == "prey":
        # Prey was the learner — checkpoint/ has prey params
        prey_params = checkpointer.restore(str(checkpoint_base / "checkpoint"))
        # Load frozen predator from opponent checkpoint if available
        opp_ckpt = configs.get("opponent_checkpoint")
        if opp_ckpt:
            opp_path = str(Path(opp_ckpt).resolve())
            pred_params = checkpointer.restore(opp_path)
        else:
            pred_params = None
    else:
        pred_params = checkpointer.restore(str(checkpoint_base / "checkpoint"))
        prey_params = None

    return pred_params, prey_params, configs["train"], env_config


def run_episode(
    pred_params,
    prey_params,
    env: PredatorPreyEnv,
    env_config: EnvConfig,
    seed: int = 0,
    max_steps: int = 500,
    prey_noise_scale: float = 0.3,
) -> list[BoidsState]:
    """Run episode with trained policies.

    If prey_params is provided, uses learned prey policy.
    Otherwise, uses random prey actions.

    Args:
        pred_params: Trained predator network parameters
        prey_params: Trained prey network parameters (None for random prey)
        env: Environment instance
        env_config: Environment configuration
        seed: Random seed
        max_steps: Maximum steps per episode
        prey_noise_scale: Scale of random prey actions (only used if prey_params is None)

    Returns:
        List of states for visualization
    """
    pred_network = ActorCritic(action_dim=env.action_size) if pred_params is not None else None
    prey_network = ActorCritic(action_dim=env.action_size) if prey_params is not None else None

    key = jax.random.PRNGKey(seed)
    _, state = env.reset(key)
    states = [state]

    for _ in range(max_steps):
        key, k1, k2, k3 = jax.random.split(key, 4)

        # Predator actions: learned policy or random
        if pred_network is not None and pred_params is not None:
            pred_obs = env.get_obs(state)["predator"]  # [n_pred, obs_size]
            pred_out = jax.vmap(lambda o: pred_network.apply(pred_params, o))(pred_obs)
            pred_pi = make_distribution(pred_out.action_mean, pred_out.action_logstd)
            pred_actions = pred_pi.sample(seed=k1)
        else:
            pred_actions = jax.random.normal(k1, (env_config.n_predators, env.action_size))
            pred_actions = pred_actions * prey_noise_scale

        # Prey actions: learned policy or random
        if prey_network is not None and prey_params is not None:
            prey_obs = env.get_obs(state)["prey"]  # [n_prey, obs_size]
            prey_out = jax.vmap(lambda o: prey_network.apply(prey_params, o))(prey_obs)
            prey_pi = make_distribution(prey_out.action_mean, prey_out.action_logstd)
            prey_actions = prey_pi.sample(seed=k2)
        else:
            prey_actions = jax.random.normal(k2, (env_config.n_prey, env.action_size))
            prey_actions = prey_actions * prey_noise_scale

        actions = {"predator": pred_actions, "prey": prey_actions}
        _, state, _, dones, _ = env.step(k3, state, actions)
        states.append(state)

        if dones["__all__"]:
            break

    return states


def main():
    parser = argparse.ArgumentParser(description="Visualize trained predator-prey agents")
    parser.add_argument(
        "run_name",
        type=str,
        help="Path to training run directory (e.g., 'runs/train_ippo_5m')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for episode (default: 0)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save animation (e.g., 'output.gif')",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=50,
        help="Animation interval in ms (default: 50)",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.run_name}...")
    pred_params, prey_params, train_config, env_config = load_checkpoint(args.run_name)

    if pred_params is not None and prey_params is not None:
        mode = "both learned"
    elif prey_params is not None:
        mode = "prey learned, predators random"
    elif pred_params is not None:
        mode = "predators learned, prey random"
    else:
        mode = "both random"
    print(f"Mode: {mode}")
    print(f"Environment: {env_config.n_predators} predators, {env_config.n_prey} prey")

    env = PredatorPreyEnv(env_config)

    prey_noise = train_config.get("prey_noise_scale", 0.3)
    print(f"Running episode with seed={args.seed}...")
    states = run_episode(
        pred_params,
        prey_params,
        env,
        env_config,
        seed=args.seed,
        max_steps=args.max_steps,
        prey_noise_scale=prey_noise,
    )
    print(f"Episode finished after {len(states)} steps")

    final_alive = int(jnp.sum(states[-1].prey_alive))
    print(f"Final prey alive: {final_alive}/{env_config.n_prey}")

    animate_episode(states, env_config, interval=args.interval, save_path=args.save)
    if args.save:
        print(f"Saved animation to {args.save}")


if __name__ == "__main__":
    main()
