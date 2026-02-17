"""Quick demo to visualize boids behavior."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.types import EnvConfig


def run_demo(n_steps: int = 200, seed: int = 42, save_path: str | None = None):
    """Run and visualize a boids episode with random actions."""
    config = EnvConfig(
        n_predators=5,
        n_prey=10,
        world_size=100.0,
        max_speed=5.0,
        perception_radius=15.0,
    )
    env = PredatorPreyEnv(config)

    key = jax.random.PRNGKey(seed)
    _, state = env.reset(key)

    # Collect trajectory
    states = [state]
    for _ in range(n_steps):
        key, k1, k2, k3 = jax.random.split(key, 4)
        actions = {
            "predator": jax.random.uniform(k1, (config.n_predators, 2), minval=-1, maxval=1),
            "prey": jax.random.uniform(k2, (config.n_prey, 2), minval=-1, maxval=1),
        }
        _, state, _, dones, _ = env.step(k3, state, actions)
        states.append(state)
        if dones["__all__"]:
            break

    # Visualize
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()
        state = states[frame]

        ax.set_xlim(0, config.world_size)
        ax.set_ylim(0, config.world_size)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")

        # Predators (red triangles)
        pred_pos = state.predator_pos
        pred_vel = state.predator_vel
        ax.scatter(pred_pos[:, 0], pred_pos[:, 1], c="red", s=100, marker="^", zorder=3)
        ax.quiver(
            pred_pos[:, 0],
            pred_pos[:, 1],
            pred_vel[:, 0],
            pred_vel[:, 1],
            color="salmon",
            scale=30,
            alpha=0.7,
            zorder=2,
        )

        # Alive prey (cyan circles)
        prey_pos = state.prey_pos
        prey_vel = state.prey_vel
        alive = state.prey_alive

        alive_pos = prey_pos[alive]
        alive_vel = prey_vel[alive]
        ax.scatter(alive_pos[:, 0], alive_pos[:, 1], c="cyan", s=80, marker="o", zorder=3)
        if len(alive_pos) > 0:
            ax.quiver(
                alive_pos[:, 0],
                alive_pos[:, 1],
                alive_vel[:, 0],
                alive_vel[:, 1],
                color="lightcyan",
                scale=30,
                alpha=0.7,
                zorder=2,
            )

        # Dead prey (gray x)
        dead_pos = prey_pos[~alive]
        if len(dead_pos) > 0:
            ax.scatter(dead_pos[:, 0], dead_pos[:, 1], c="gray", s=40, marker="x", alpha=0.5)

        ax.set_title(
            f"Step {state.step} | Prey alive: {int(jnp.sum(alive))}/{config.n_prey}",
            color="white",
            fontsize=12,
        )

        return []

    anim = FuncAnimation(fig, update, frames=len(states), interval=50, blit=False)

    if save_path:
        anim.save(save_path, writer="pillow", fps=20)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import sys

    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_demo(save_path=save_path)
