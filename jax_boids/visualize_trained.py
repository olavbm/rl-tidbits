"""Visualize a trained predator agent."""

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


def load_checkpoint(run_name: str, runs_dir: str = ""):
    """Load params and configs from a training run.

    Args:
        run_name: Path to run directory (e.g., 'runs/long_training/validated_005_seed124/pred_vs_random_20260313_164350')
        runs_dir: Additional prefix directory (usually empty)

    Returns:
        params: Network parameters
        train_config: Training configuration dict
        env_config: Environment configuration as EnvConfig
    """
    run_dir = Path(run_name).resolve()

    # Load params with Orbax (requires absolute paths)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpoint_dir = run_dir / "checkpoint"
    params = checkpointer.restore(str(checkpoint_dir))

    # Load configs from JSON
    config_path = run_dir / "config.json"
    with open(config_path) as f:
        configs = json.load(f)

    # Reconstruct EnvConfig
    env_config = EnvConfig(**configs["env"])

    return params, configs["train"], env_config


def run_trained_episode(
    params,
    env: PredatorPreyEnv,
    env_config: EnvConfig,
    seed: int = 0,
    max_steps: int = 500,
    prey_noise_scale: float = 0.3,
) -> list[BoidsState]:
    """Run episode with trained predator policy, random prey.

    Args:
        params: Trained network parameters
        env: Environment instance
        env_config: Environment configuration
        seed: Random seed
        max_steps: Maximum steps per episode
        prey_noise_scale: Scale of random prey actions

    Returns:
        List of states for visualization
    """
    network = ActorCritic(action_dim=env.action_size)
    key = jax.random.PRNGKey(seed)

    _, state = env.reset(key)
    states = [state]

    for _ in range(max_steps):
        key, k1, k2, k3 = jax.random.split(key, 4)

        # Get predator observations and actions from trained policy
        pred_obs = env.get_obs(state)["predator"]  # [n_pred, obs_size]
        pred_out = jax.vmap(lambda o: network.apply(params, o))(pred_obs)
        pred_pi = make_distribution(pred_out.action_mean, pred_out.action_logstd)
        pred_actions = pred_pi.sample(seed=k1)

        # Random prey actions
        prey_actions = jax.random.normal(k2, (env_config.n_prey, env.action_size))
        prey_actions = prey_actions * prey_noise_scale

        actions = {"predator": pred_actions, "prey": prey_actions}
        _, state, _, dones, _ = env.step(k3, state, actions)
        states.append(state)

        if dones["__all__"]:
            break

    return states


def main():
    parser = argparse.ArgumentParser(description="Visualize a trained predator agent")
    parser.add_argument(
        "run_name",
        type=str,
        help="Name of the training run (e.g., 'pred_vs_random_20260209_115130')",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Directory containing run folders (default: runs)",
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
    parser.add_argument(
        "--max-speed",
        type=float,
        default=1.0,
        help="Maximum speed for all agents (default: 1.0)",
    )
    args = parser.parse_args()

    # Handle path: if run_name already contains a path, use it directly
    checkpoint_path = args.run_name if "/" in args.run_name else f"{args.runs_dir}/{args.run_name}"
    print(f"Loading checkpoint from {checkpoint_path}...")
    params, train_config, env_config = load_checkpoint(checkpoint_path)

    prey_noise_scale = train_config.get("prey_noise_scale", 0.3)
    print(f"Environment: {env_config.n_predators} predators, {env_config.n_prey} prey")
    print(f"Prey noise scale: {prey_noise_scale}")

    env = PredatorPreyEnv(env_config)

    print(f"Running episode with seed={args.seed}...")
    states = run_trained_episode(
        params,
        env,
        env_config,
        seed=args.seed,
        max_steps=args.max_steps,
        prey_noise_scale=prey_noise_scale,
    )
    print(f"Episode finished after {len(states)} steps")

    final_alive = int(jnp.sum(states[-1].prey_alive))
    print(f"Final prey alive: {final_alive}/{env_config.n_prey}")

    animate_episode(states, env_config, interval=args.interval, save_path=args.save)
    if args.save:
        print(f"Saved animation to {args.save}")


if __name__ == "__main__":
    main()
