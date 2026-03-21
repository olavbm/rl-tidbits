# AGENTS.md

## Project Overview

RL-tidbits: reinforcement learning experiments built with JAX and Gymnasium.

## Active Project: JAX-based IPPO for Multi-Agent RL

JAX implementation of Independent PPO (IPPO) for multi-agent predator-prey in `jax_boids/`.

### Goal

Reduce prey_alive to <1.5 with two-sided IPPO (both predators and prey learn, no boids forces).

### Status

- **IPPO active**: `best_pred2` config, prey_alive=2.37 at 5M steps (both sides learning)
- Predator-only baseline: prey_alive=2.17 at 1M steps (`sweep_r03_003`, prey were passive boids)
- Key problem: predators stack on top of each other due to shared weights + identical observations

### Environment (defaults / IPPO setup)

- Default: 1 predator, 3 prey. IPPO configs use 2 predators, 5 prey.
- 10x10 grid, 200 steps per episode
- Predators share network weights (parameter sharing)
- Physics: max_speed=1.0, max_acceleration=0.5, velocity_damping=0.9
- predator_speed_bonus=1.2, prey_speed_mult=0.5 (2.4× speed advantage)
- Capture radius: 0.3
- Observation: k_nearest_same=4, k_nearest_enemy=3 (30-dim obs)

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

1. **Add agent index to observations** — Both predators share weights and see identical relative geometry when close, producing identical actions → stacking. Adding a unique index (e.g., `[0.0]` vs `[1.0]`) to each agent's obs lets the shared network differentiate them. Standard fix for parameter-sharing MARL. Minimal code change in `_single_agent_obs`. (30-dim → 31-dim)

2. **Richer observations** — Consider adding normalized own position so agents can learn spatial strategies. k_nearest_enemy=3 already covers 3 of 5 prey; could increase to 4-5 for full visibility.

3. **Anti-stacking reward shaping** — Small bonus proportional to inter-predator distance. Directly incentivizes spreading out. Complements agent index approach.

4. **Larger network** — Current 64×64 MLP may lack capacity for 2v5 IPPO coordination. Try 128×128 or add a third layer.

## Deployment

Training runs execute on remote machine (hoppetusse) via `deploy.sh`:

```bash
# Sweep
./deploy.sh jax_boids/train.py --mode sweep --n-configs 100 --total-timesteps 1000000

# Train named config
./deploy.sh jax_boids/train.py --mode train --config validated_005

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
