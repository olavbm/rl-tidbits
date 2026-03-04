# AGENTS.md

This file provides guidance for AI coding assistants working with this repository.

## Project Overview

RL-tidbits is a collection of reinforcement learning experiments and agents built with Gymnasium.

## Active Project: JAX-based IPPO for Multi-Agent RL

Currently working on a JAX implementation of Independent PPO (IPPO) for multi-agent reinforcement learning in the `jax_boids/` directory.

### Current Status

**Debugging value function instability in IPPO implementation:**
- Implementation is fundamentally correct (GAE, PPO clipping, advantage normalization all validated)
- Policy learns but value function predictions are unstable (predictions ~0.2 vs actual returns ~11.7)
- Value loss fluctuates wildly during training (19-207 range)
- Running extended training tests to determine if value function eventually converges

### Environment

- Predator-prey domain: 2 predators, 2 prey, 30x30 world
- Heterogeneous agents with separate networks per agent type
- Standard PPO hyperparameters: γ=0.99, λ=0.95, clip=0.2, lr=3e-4

### Key Files

- `jax_boids/ppo.py` - Core PPO implementation (GAE, loss, update)
- `jax_boids/networks.py` - ActorCritic network
- `jax_boids/train.py` - Multi-agent training script
- `jax_boids/collector.py` - Rollout collection
- `jax_boids/diagnose.py` - Diagnostic script for debugging

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
