# AGENTS.md

This file provides guidance for AI coding assistants working with this repository.

## Project Overview

RL-tidbits is a collection of reinforcement learning experiments and agents built with Gymnasium.

## Active Project: JAX-based IPPO for Multi-Agent RL

JAX implementation of Independent PPO (IPPO) for multi-agent reinforcement learning in `jax_boids/`.

### Current Goal

Improve PPO training efficiency for predator-prey environment. Target: reduce prey_alive from ~2.5 to <1.5 before enabling two-sided learning.

### Current Status

**Expanded random hyperparameter sweep completed:**
- 100 configs on simple env (1 predator, 3 prey, 10x10, 100 steps)
- **7 configs achieved target prey_alive < 1.5** (best: 1.415)
- Results in `runs/expanded_random_sweep/`
- Best config: trial_099 (lr=1.3e-3, clip=0.22, ent=0.091, n_steps=64, n_epochs=2, ortho=True, lr_anneal=True, norm_ret=True)

**Next steps:**
- Validate top configs with extended training
- Fine-tune around best hyperparameters
- Test on larger environments (5v5)

**PPO features implemented:**
- Orthogonal initialization (`networks.py`)
- LR annealing (`ppo.py`)
- Adam eps=1e-5 (always on)
- Return normalization (`ppo.py`)
- All controlled via `TrainConfig` flags in `train_single.py`

### Environment

- Predator-prey domain with configurable agents and world size
- Heterogeneous agents with separate networks per agent type
- Default PPO hyperparameters: γ=0.99, λ=0.95, clip=0.2, lr=3e-4, max_grad_norm=0.5

### Key Files

- `jax_boids/ppo.py` - Core PPO implementation
- `jax_boids/networks.py` - ActorCritic network
- `jax_boids/train_single.py` - Single-agent training (defaults: ortho=True, anneal=True, norm_ret=True)
- `jax_boids/run_random_sweep.py` - Random hyperparameter sweep (broad search)
- `jax_boids/run_fine_tuning_sweep.py` - Fine-tuning sweep (tight bounds around best)
- `jax_boids/analyze_sweep.py` - Sweep results analysis
- `jax_boids/test_original_env.py` - Test configs on 5v5 environment
- `jax_boids/collector.py` - Rollout collection
- `jax_boids/telemetry/diagnostics.py` - Training diagnostics

## Deployment

All training runs should be executed on the remote machine (hoppetusse) via `deploy.sh`:

### Running Sweeps

```bash
# Broad random sweep
./deploy.sh jax_boids/run_random_sweep.py --n-configs 100

# Fine-tuning sweep (tight bounds around best config)
./deploy.sh jax_boids/run_fine_tuning_sweep.py --n-configs 100

# Check progress
ssh hoppetusse "tail -f dev/python/rl-tidbits/run_fine_tuning_sweep_output.log"

# Fetch results when done
./deploy.sh --fetch runs/fine_tuning_sweep

# Analyze results
./deploy.sh --analyze runs/fine_tuning_sweep/results.json --top 10
```

### Running Single Training

```bash
# Deploy training script
./deploy.sh jax_boids/train_single.py

# Check progress
ssh hoppetusse "tail -f dev/python/rl-tidbits/train_single_output.log"
```

The deploy script:
1. Syncs code to remote (excludes `.git`, `.venv`, `runs`, `__pycache__`)
2. Kills any existing process running the same script
3. Starts training in background with `nohup`
4. Logs output to `<script_name>_output.log`

### Analysis Tools

```bash
# Analyze sweep results (JSON file or directory with results.json)
./deploy.sh --analyze runs/expanded_random_sweep/results.json --top 10

# Compare multiple training runs
uv run python -m jax_boids.telemetry.diagnostics runs/<sweep_dir> --compare --agent pred
```

## Commands

```bash
# Install and activate (using uv)
uv sync
source .venv/bin/activate

# Run linter
ruff check .
ruff check . --fix

# Format code
ruff format .

# Run tests
pytest
```

## RL Conventions

- Use Gymnasium (not legacy Gym) for environments
- Agents should implement `select_action(observation)` and `learn(...)` methods
- Use NumPy for numerical operations