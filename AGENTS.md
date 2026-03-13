# AGENTS.md

This file provides guidance for AI coding assistants working with this repository.

## Project Overview

RL-tidbits is a collection of reinforcement learning experiments and agents built with Gymnasium.

## Active Project: JAX-based IPPO for Multi-Agent RL

JAX implementation of Independent PPO (IPPO) for multi-agent reinforcement learning in `jax_boids/`.

### Current Goal

Improve PPO training efficiency for predator-prey environment. Target: reduce prey_alive from ~2.5 to <1.5 before enabling two-sided learning.

### Current Status

**Completed:**
- Expanded random sweep (100 configs, 1M steps): 7 configs achieved prey_alive < 1.5
- Two rounds of validation (2M steps): Only 2 configs consistently validated
- Config database created with named validated configs

**Key findings:**
- Original best (trial_099, 1.415) failed validation at 1.500 - high variance
- Only `validated_005` and `validated_043` reliably achieve ~1.500 prey_alive
- Both use low learning rates (~1e-4), 128 n_steps, 10 n_epochs, no LR annealing
- Validation is reproducible (8/10 configs identical between rounds)

**Next steps:**
- Multi-seed validation on validated_005/validated_043 to confirm robustness
- Test on 5v5 environment
- Fine-tune around validated hyperparameters if needed

**PPO features implemented:**
- Orthogonal initialization (`networks.py`)
- LR annealing (`ppo.py`)
- Adam eps=1e-5 (always on)
- Return normalization (`ppo.py`)

### Environment

- Predator-prey domain: 1 predator, 3 prey, 10x10 grid, 100 steps
- Heterogeneous agents with separate networks per agent type
- Default PPO: γ=0.99, λ=0.95, clip=0.2, lr=3e-4, max_grad_norm=0.5

### Key Files

- `jax_boids/configs.py` - Named config database (validated_005, validated_043)
- `jax_boids/validate_configs.py` - Extended validation script (2M steps)
- `jax_boids/ppo.py` - Core PPO implementation
- `jax_boids/train_single.py` - Single-agent training (predators learn)
- `jax_boids/analyze_sweep.py` - Sweep results analysis
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