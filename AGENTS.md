# AGENTS.md

## Project Overview

RL-tidbits: reinforcement learning experiments built with JAX and Gymnasium.

## Active Project: JAX-based IPPO for Multi-Agent RL

JAX implementation of Independent PPO (IPPO) for multi-agent predator-prey in `jax_boids/`.

### Goal

Reduce prey_alive from ~2.5 to <1.5 (predator-only learning), then enable two-sided IPPO.

### Status

- Large sweep running on hoppetusse: 100 configs, 1M timesteps each (`runs/big_sweep_1m/`)
- Previous validated configs: `validated_005`, `validated_043` (both ~1.500 prey_alive)
- GPU utilization is low when llama-server occupies GPU memory on hoppetusse

### Environment

- Predator-prey: 1 predator, 3 prey, 10x10 grid, 100 steps
- Heterogeneous agents with separate networks per agent type
- Physics: max_speed=1.0, max_acceleration=0.5, velocity_damping=0.9

### Key Files

- `jax_boids/train.py` - Unified training entrypoint (train, sweep, sweep-fine, validate)
- `jax_boids/train_single.py` - Single-agent training loop (predators learn, prey random)
- `jax_boids/train_ippo.py` - Multi-agent IPPO training (both sides learn)
- `jax_boids/ppo.py` - Core PPO implementation
- `jax_boids/envs/types.py` - `TrainConfig`, `EnvConfig`, `BoidsState`
- `jax_boids/configs.py` - Named config database
- `jax_boids/analyze_sweep.py` - Sweep results analysis

### Architecture Notes

- `make_train()` in `train_single.py` returns `(jit_step_fn, steps_per_update)` — no callbacks inside the JIT-compiled step
- `train()` runs steps in batches via `jax.lax.scan(step_fn, state, None, length=batch_size)`
- All TensorBoard logging and checkpointing happens after training completes (no CPU-GPU sync during training)

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
