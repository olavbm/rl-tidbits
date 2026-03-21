# AGENTS.md

## Project Overview

RL-tidbits: reinforcement learning experiments built with JAX and Gymnasium.

## Active Project: JAX-based IPPO for Multi-Agent RL

JAX implementation of Independent PPO (IPPO) for multi-agent predator-prey in `jax_boids/`.

### Goal

Reduce prey_alive to <1.5 with two-sided IPPO (both predators and prey learn, no boids forces).

### Status

- **Large-scale IPPO**: 5 predators vs 100 prey, 50x50 world, 1000 steps/episode
  - With agent index: prey_alive=36.5 at 50M steps (64 of 100 prey caught)
  - Without agent index: prey_alive=46.0 at 50M steps
- **Small-scale IPPO** (2v5, 10x10): `best_pred2` config, prey_alive=2.37 at 5M steps
- Agent index in observations resolved predator clumping from shared weights

### Environment

- Default: 1 predator, 3 prey, 10x10 grid, 200 steps
- All env params configurable via CLI (k_nearest_same, k_nearest_enemy, world_size, etc.)
- Predators share network weights (parameter sharing)
- Each agent gets a normalized index in its observation for differentiation
- Physics: max_speed=1.0, max_acceleration=0.5, velocity_damping=0.9
- Observation: own velocity + agent index + k-nearest relative positions/velocities

### Key Files

- `jax_boids/train.py` - Unified training entrypoint (train, sweep, sweep-fine, validate)
- `jax_boids/train_single.py` - Single-agent training loop (one side learns, other random/frozen)
- `jax_boids/train_ippo.py` - Multi-agent IPPO training (both sides learn)
- `jax_boids/collector.py` - Rollout collection (separates env interaction from training)
- `jax_boids/ppo.py` - Core PPO implementation
- `jax_boids/networks.py` - Actor-critic network (64×64 MLP)
- `jax_boids/envs/types.py` - `TrainConfig`, `EnvConfig`, `BoidsState`
- `jax_boids/envs/predator_prey.py` - Environment logic, observations, rewards
- `jax_boids/configs.py` - Named config database
- `jax_boids/visualize_trained.py` - Checkpoint loading and episode visualization
- `jax_boids/analyze_sweep.py` - Sweep results analysis

### Architecture Notes

- `make_train()` returns `(train_fn, steps_per_update)` — pure function, no side effects
- `get_train_fn()` wraps `make_train()` with `jax.jit` and caches by shape-affecting params
- Scalar hyperparams (lr, clip_eps, etc.) passed as JAX arrays via `ScalarHParams` to avoid recompilation
- Training loop runs via `jax.lax.scan(_update_step, ..., length=n_updates)` — no CPU-GPU sync
- All TensorBoard logging and checkpointing happens after training completes (single sync point)

### Next Steps (prioritized)

1. **Richer observations** — Add normalized own position so agents can learn spatial strategies (e.g. territory assignment).

2. **Anti-stacking reward shaping** — Small bonus proportional to inter-predator distance. Directly incentivizes spreading out, complementing agent index.

3. **Larger network** — Current 64×64 MLP may lack capacity for large-scale (5v100) coordination. Try 128×128 or add a third layer.

4. **More training** — Large-scale runs may benefit from 100M+ steps to fully converge.

## Deployment

Training runs execute on remote machine (hoppetusse) via `deploy.sh`:

```bash
# Sweep
./deploy.sh jax_boids/train.py --mode sweep --n-configs 100 --total-timesteps 1000000

# Train named config
./deploy.sh jax_boids/train.py --mode train --config validated_005

# Large-scale IPPO (5 predators, 100 prey, 50x50 world)
./deploy.sh jax_boids/train.py --mode train --config best_pred2 --prey-learn \
  --n-predators 5 --n-prey 100 --world-size 50 --max-steps 1000 \
  --capture-radius 1.5 --predator-speed-bonus 2.0 \
  --k-nearest-same 6 --k-nearest-enemy 20 --total-timesteps 50000000 --verbose

# Validate with multiple seeds
./deploy.sh jax_boids/train.py --mode validate --config validated_005 --n-seeds 5

# Validate top N from sweep
./deploy.sh jax_boids/train.py --mode validate --from-results runs/sweep/results.json --top 10

# Check progress / fetch results / analyze
ssh hoppetusse "tail -f dev/python/rl-tidbits/train_output.log"
./deploy.sh --fetch runs/sweep
./deploy.sh --analyze runs/sweep/results.json --top 10
```

## Commands

```bash
uv sync                  # Install dependencies
uv run ruff check .      # Lint
uv run ruff format .     # Format
uv run pytest            # Test
```
