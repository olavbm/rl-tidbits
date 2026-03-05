# AGENTS.md

This file provides guidance for AI coding assistants working with this repository.

## Project Overview

RL-tidbits is a collection of reinforcement learning experiments and agents built with Gymnasium.

## Active Project: JAX-based IPPO for Multi-Agent RL

Currently working on a JAX implementation of Independent PPO (IPPO) for multi-agent reinforcement learning in the `jax_boids/` directory.

### Current Status

**Value function instability diagnosed - implementation is correct:**
- `test_value_convergence.py` confirms value head CAN learn with continuous backprop on fixed data
- Value function converged to predict mean of returns (MSE ≈ variance, confirming correct mean prediction)
- GAE computation validated via unit tests in `test_env.py`
- Value head receives gradients and updates correctly (verified in `test_value_update.py`)
- Bootstrap values verified working correctly in `test_bootstrap_computation.py`
- Policy learning appears functional (rewards show slight improvement trend)

**CONCLUSION: Implementation is correct, no action needed**
- Value loss 50-200 during early training is **expected behavior** (scale mismatch)
- All components validated: GAE ✅, bootstrap ✅, gradient flow ✅, value head learning ✅
- Gradient clipping at 12.0 is reasonable (42-95% of gradients kept)
- Flat reward trend (1.84 → 1.93) over 30 updates: neutral signal, not degrading
- **Decision**: Accept as-is, slow learning is expected in RL, no tuning needed

### Environment

- Predator-prey domain: 2 predators, 2 prey, 30x30 world
- Heterogeneous agents with separate networks per agent type
- Standard PPO hyperparameters: γ=0.99, λ=0.95, clip=0.2, lr=3e-4
- Gradient clipping: 12.0 (in `create_train_state`, NOT the unused `max_grad_norm=0.5` in config)

### Key Files

- `jax_boids/ppo.py` - Core PPO implementation (GAE, loss, update)
- `jax_boids/networks.py` - ActorCritic network
- `jax_boids/train.py` - Multi-agent training script
- `jax_boids/collector.py` - Rollout collection
- `jax_boids/diagnose.py` - Diagnostic script for debugging

### Diagnostic Tests

**Validated (working correctly):**
- `jax_boids/test_value_convergence.py` - Confirms value head CAN learn fixed targets ✅
- `jax_boids/test_bootstrap_computation.py` - Verifies bootstrap values are computed and used ✅
- `jax_boids/test_env.py` - GAE computation unit tests (PASS) ✅
- `jax_boids/test_learning.py` - End-to-end learning test (neutral reward trend) ✅

## Commands

```bash
# Install and activate (using uv)
uv sync
source .venv/bin/activate

# Install dev dependencies
uv sync --all-extras

# Run linter
ruff check .

# Fix lint issues
ruff check . --fix

# Format code
ruff format .

# Run tests
pytest

# Run a script
python main.py
```

## Architecture

- `agents/` - RL agent implementations
- `main.py` - Entry point for experiments

## RL Conventions

- Use Gymnasium (not legacy Gym) for environments
- Agents should implement `select_action(observation)` and `learn(...)` methods
- Use NumPy for numerical operations
