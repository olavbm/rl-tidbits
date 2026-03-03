# AGENTS.md

This file provides guidance for AI coding assistants working with this repository.

## Project Overview

RL-tidbits is a collection of reinforcement learning experiments and agents built with Gymnasium.

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
