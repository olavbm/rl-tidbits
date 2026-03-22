# rl-tidbits

Reinforcement learning experiments in JAX.

## Predator-Prey with IPPO

Multi-agent predator-prey on a toroidal grid, trained with Independent PPO (IPPO). Both predators and prey learn simultaneously. Everything is JAX — environment, PPO, training loop — compiled into a single `jax.lax.scan` with no CPU-GPU sync during training.

![IPPO predator-prey](media/ippo_100m.gif)

*5 predators (red) vs 100 prey (blue), both sides learning via IPPO on a 50x50 world.*

### Current state

Best result: **prey_alive=33.2** at 50M steps (5v100, LR=0.0013, 8 epochs). At 100M steps, an arms race regression brought this to 44.2 as prey learned better evasion strategies.

Key features of the current implementation:
- **Parameter sharing** with normalized agent index for differentiation
- **Asymmetric networks**: predators 64x64x64, prey 64x64
- **Observations**: own velocity, own position (normalized), agent index, k-nearest same-team and enemy relative positions/velocities
- **Dead prey**: velocity zeroed on capture, disappear from simulation
- **Proper logging** via Python `logging` module

### Where it left off

The core IPPO loop works well. The main open problem is the **arms race regression** — prey eventually learn evasion strategies that undo predator progress. See `docs/marl_research.md` for a detailed analysis of next steps, ranked by expected impact:

1. **Asymmetric training** — alternate which side trains to decouple learning dynamics
2. **Reward shaping** — individual capture credit + anti-stacking penalty for better coordination
3. **Opponent sampling** — train against a pool of past checkpoints to prevent cycling
4. **MAPPO** — centralized critic for predators (sees global state during training)

Architecture-level prep work is planned in `AGENTS.md`. The codebase is structured for volatility-based decomposition — algorithms and networks should be swappable without rewriting training loops.

### Running

```bash
uv sync

# Train with IPPO (both sides learn)
uv run python -m jax_boids.train --mode train --config best_pred2 --prey-learn

# Large-scale (5v100)
uv run python -m jax_boids.train --mode train --config best_pred2 --prey-learn \
  --n-predators 5 --n-prey 100 --world-size 50 --max-steps 1000 \
  --capture-radius 1.5 --predator-speed-bonus 2.0 \
  --k-nearest-same 6 --k-nearest-enemy 20 --total-timesteps 50000000

# Hyperparameter sweep
uv run python -m jax_boids.train --mode sweep --n-configs 100 --total-timesteps 1000000

# Visualize a trained checkpoint
uv run python -m jax_boids.visualize_trained runs/<run_dir>

# Tests
uv run pytest
```

### Structure

```
jax_boids/
  envs/              # Predator-prey environment (pure JAX)
    predator_prey.py # Environment logic, observations, rewards
    rewards.py       # Reward functions (capture, distance, prey survival)
    types.py         # EnvConfig, TrainConfig, BoidsState
    boids.py         # Boids physics (separation, alignment, cohesion)
  networks.py        # Actor-critic MLP (configurable hidden_dims)
  ppo.py             # Core PPO (GAE, loss, update)
  collector.py       # Rollout collection (separates env interaction from training)
  train.py           # Unified CLI entrypoint (train, sweep, validate)
  train_ippo.py      # IPPO training (both sides learn)
  train_single.py    # Single-agent training (one side learns)
  configs.py         # Named hyperparameter configs
  visualize.py       # Episode rendering to GIF
  visualize_trained.py # Load checkpoint and visualize
  analyze_run.py     # Convergence analysis for training runs
docs/
  marl_research.md   # MARL methods analysis and next steps
```

### Deployment

Training runs on a remote GPU machine via `deploy.sh`:

```bash
./deploy.sh jax_boids/train.py --mode train --config best_pred2 --prey-learn ...
./deploy.sh --fetch runs/<run_dir>    # Fetch results
```

## Past experiments

- **Humanoid-v5** — SAC with stable-baselines3 and Gymnasium (code removed, in git history)
