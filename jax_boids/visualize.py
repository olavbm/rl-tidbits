"""Visualization for boids environment."""

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.types import BoidsState, EnvConfig


def render_state(
    ax: plt.Axes,
    state: BoidsState,
    config: EnvConfig,
    show_velocity: bool = True,
):
    """Render a single frame."""
    ax.clear()
    ax.set_xlim(0, config.world_size)
    ax.set_ylim(0, config.world_size)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")

    # Draw predators (red) with capture radius
    pred_pos = state.predator_pos
    pred_vel = state.predator_vel
    for i in range(len(pred_pos)):
        circle = mpatches.Circle(
            (float(pred_pos[i, 0]), float(pred_pos[i, 1])),
            config.capture_radius,
            color="red",
            fill=False,
            linestyle="--",
            alpha=0.3,
            linewidth=1,
            zorder=1,
        )
        ax.add_patch(circle)
    ax.scatter(
        pred_pos[:, 0], pred_pos[:, 1], c="red", s=80, marker="^", label="Predators", zorder=3
    )
    if show_velocity:
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

    # Draw prey (blue, only alive ones)
    prey_pos = state.prey_pos
    prey_vel = state.prey_vel
    alive = state.prey_alive

    alive_pos = prey_pos[alive]
    alive_vel = prey_vel[alive]
    dead_pos = prey_pos[~alive]

    ax.scatter(
        alive_pos[:, 0], alive_pos[:, 1], c="cyan", s=60, marker="o", label="Prey (alive)", zorder=3
    )
    if show_velocity and len(alive_pos) > 0:
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

    # Draw dead prey (gray, faded)
    if len(dead_pos) > 0:
        ax.scatter(
            dead_pos[:, 0],
            dead_pos[:, 1],
            c="gray",
            s=30,
            marker="x",
            alpha=0.5,
            label="Prey (dead)",
            zorder=1,
        )

    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Step {state.step} | Prey alive: {int(jnp.sum(alive))}/{config.n_prey}")


def run_episode_random(config: EnvConfig, seed: int = 0, max_steps: int = 200):
    """Run episode with random actions and return states."""
    env = PredatorPreyEnv(config)
    key = jax.random.PRNGKey(seed)

    _, state = env.reset(key)
    states = [state]

    for _ in range(max_steps):
        key, k1, k2, k3 = jax.random.split(key, 4)

        # Random actions
        pred_actions = jax.random.uniform(k1, (config.n_predators, 2), minval=-1, maxval=1)
        prey_actions = jax.random.uniform(k2, (config.n_prey, 2), minval=-1, maxval=1)
        actions = {"predator": pred_actions, "prey": prey_actions}

        _, state, _, dones, _ = env.step(k3, state, actions)
        states.append(state)

        if dones["__all__"]:
            break

    return states


def animate_episode(
    states: list[BoidsState],
    config: EnvConfig,
    interval: int = 50,
    save_path: str | None = None,
):
    """Create animation from episode states."""
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        render_state(ax, states[frame], config)
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(states), interval=interval, blit=False)

    if save_path:
        anim.save(save_path, writer="pillow", fps=1000 // interval)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()

    return anim


def visualize_random_episode(config: EnvConfig | None = None, seed: int = 0):
    """Quick visualization of random policy."""
    config = config or EnvConfig()
    print("Running episode with random actions...")
    states = run_episode_random(config, seed)
    print(f"Episode finished after {len(states)} steps")
    animate_episode(states, config)


if __name__ == "__main__":
    visualize_random_episode()
