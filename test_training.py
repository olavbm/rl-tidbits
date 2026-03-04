"""Quick test to observe value function instability."""

import jax
from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.types import EnvConfig
from jax_boids.train import TrainConfig, RunnerState, make_train
from jax_boids.ppo import create_train_state

# Small config for quick testing
config = TrainConfig(
    total_timesteps=100_000,  # Longer training
    n_envs=32,
    n_steps=128,
)
env_config = EnvConfig(n_predators=2, n_prey=2)
env = PredatorPreyEnv(env_config)


def log_fn(step, metrics):
    print(
        f"Step {step:6d} | "
        f"pred_vloss={metrics['pred_value_loss']:7.2f} "
        f"prey_vloss={metrics['prey_value_loss']:7.2f} | "
        f"pred_r={metrics['pred_reward']:5.2f} "
        f"prey_r={metrics['prey_reward']:5.2f}"
    )


print("Compiling...")
train_fn = make_train(config, env, env_config, log_fn=log_fn)
train_fn = jax.jit(train_fn)

print("Training...")
key = jax.random.PRNGKey(42)
runner_state, metrics = train_fn(key)

print("\nSummary:")
print(f"Final pred value_loss: {metrics['pred_value_loss'][-1]:.2f}")
print(f"Final prey value_loss: {metrics['prey_value_loss'][-1]:.2f}")
print(
    f"Pred value_loss range: [{metrics['pred_value_loss'].min():.2f}, {metrics['pred_value_loss'].max():.2f}]"
)
print(
    f"Prey value_loss range: [{metrics['prey_value_loss'].min():.2f}, {metrics['prey_value_loss'].max():.2f}]"
)
