# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL-tidbits is a collection of reinforcement learning experiments and agents built with Gymnasium.

## Commands

```bash
# Create mamba environment
mamba env create -f environment.yml

# Activate environment
mamba activate rl-tidbits

# Update environment after changing environment.yml
mamba env update -f environment.yml --prune

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
