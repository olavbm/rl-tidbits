# Multi-Agent RL Research Notes

Context: JAX-based predator-prey with IPPO. 5 predators (shared weights, 64x64x64 MLP) vs 100 prey (shared weights, 64x64 MLP) on 50x50 toroidal grid, 1000 steps/episode. Both sides learn. Current best: prey_alive=33.2 at 50M steps. Arms race regression at 100M steps (prey_alive=44.2).

---

## 1. MAPPO vs IPPO

### What it is

IPPO (Independent PPO) treats each agent as an independent learner: each agent optimizes its own policy using only its local observations. Agents share network weights (with agent index for differentiation), but the critic only sees local information.

MAPPO (Multi-Agent PPO, Yu et al. 2022 - "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games") extends this with a **centralized value function** during training. The key idea is Centralized Training with Decentralized Execution (CTDE): the policy (actor) still conditions only on local observations, but the value function (critic) receives global state or concatenated observations from all agents. At execution time, only the actor is used, so no extra communication is needed.

Concretely for our setup, this means:
- **Actor**: unchanged. Each predator still outputs actions from its local obs (own vel, own pos, agent_id, k-nearest teammates, k-nearest enemies).
- **Critic**: receives additional information. Options range from concatenating all predator observations to receiving the full state vector (all positions and velocities). The critic head becomes a separate network from the actor.

Yu et al. found MAPPO matched or exceeded QMIX, MADDPG, and other specialized MARL algorithms across StarCraft Multi-Agent Challenge (SMAC), Hanabi, and MPE benchmarks. Their key findings:
- Input representation matters more than the algorithm: feeding global state to the critic significantly helped.
- PPO's clipping mechanism provides stability in the non-stationary multi-agent setting.
- Parameter sharing across agents (which we already do) was consistently beneficial.
- Value normalization (PopArt or simple running stats) was important for training stability.

### Applicability to our setup

High. We already have shared weights and IPPO. The upgrade path is well-defined:

1. Split `ActorCritic` into separate `Actor` and `Critic` modules. The actor keeps the current architecture. The critic takes concatenated global state.
2. The global state for predator critics would be: all 5 predator positions/velocities + positions/velocities of k-nearest prey (from each predator's perspective, or a global summary). For 5 predators observing 3 nearest enemies each, the naive concatenation is 5 * obs_size = 5 * 33 = 165 features, though a more compact global state representation is preferable.
3. Modify `ppo_update` to pass different inputs to actor vs critic forward passes.

One concern: with 100 prey, the global state is large. A practical approach is to give the critic access to **all 5 predator observations concatenated** plus aggregate prey statistics (count alive, mean distance to nearest predator), rather than all 100 prey states.

### Implementation difficulty

**Medium.** The main changes are:
- New `Critic` network that takes global state input (separate from `ActorCritic`).
- Modify `collect_rollouts` to also gather and store global state alongside local obs.
- Modify `ppo_update` to use local obs for actor loss, global state for value loss.
- The `Transition` NamedTuple needs a `global_state` field.

The JAX/Flax infrastructure is already in place. The hardest part is deciding what goes into the global state and making sure shapes work through `jax.lax.scan`.

### Expected benefit

**Moderate to high.** The centralized critic should improve coordination because the value function can reason about what all predators are doing simultaneously. This directly helps with the "predator stacking" problem (multiple predators chasing the same prey) because the critic can learn that states where predators are spread out have higher value. The policy then implicitly learns to spread out because the advantage estimates become more informative.

Literature suggests 10-30% improvement in cooperative tasks. For our arms race problem specifically, MAPPO alone won't solve it - it improves coordination on the predator side but doesn't address the co-learning instability.

---

## 2. Population-Based Training and Self-Play

### What it is

**Population-Based Training (PBT)** (Jaderberg et al. 2017) runs a population of agents in parallel, each with different hyperparameters. Periodically, poorly performing agents copy the weights and hyperparameters of better-performing agents, then perturb the hyperparameters. This combines random search (exploration of hyperparameters) with online optimization (exploitation via weight copying).

**Self-play** variants used in competitive settings:

- **Naive self-play**: always train against the latest opponent. Suffers from forgetting (agent beats current opponent but loses to earlier strategies). This is essentially what we have now, and it explains the arms race regression.
- **Fictitious self-play (FSP)**: train against a uniform mixture of all past policies. Converges to Nash equilibrium but is expensive (must store and sample from all historical checkpoints).
- **Prioritized FSP (PFSP)** (AlphaStar, Vinyals et al. 2019): prioritize opponents that the current agent struggles against. Used in AlphaStar's league training with three agent types: main agents, league exploiters, and main exploiters.
- **OpenAI Five** (Berner et al. 2019): used large-scale self-play with "surgery" (changing network architecture during training) and reward shaping schedules. 80% of their reward was from handcrafted shaping. Key finding: team reward (shared) worked better than individual reward for Dota 2.

**How these address the arms race:**

The core problem is non-transitivity: predator strategy A beats prey strategy B, but prey strategy C beats predator strategy A, and predator strategy D beats prey strategy C but loses to prey strategy B. Naive self-play cycles through these. FSP/PFSP solve this by maintaining diversity in the opponent pool, forcing agents to be robust against a range of strategies rather than over-fitting to the latest opponent.

For our setup, a practical approach:
1. Checkpoint predator and prey policies every N updates.
2. When training predators, sample the prey opponent from a pool of recent checkpoints (not just the latest).
3. Vice versa for prey training.
4. Optionally maintain a "best response" metric: which checkpoint pool opponent does the current agent struggle with most? Prioritize that opponent.

**PBT for hyperparameter tuning** is separate from self-play. We've already done extensive sweeps (see `configs.py`). PBT could replace offline sweeps with online adaptation - especially useful for the co-learning setting where optimal hyperparameters shift as the opponent changes. For instance, predators might need higher entropy early (to explore coordination strategies) and lower entropy later (to exploit learned coordination).

### Applicability to our setup

**High, and directly addresses the arms race problem.** The regression from prey_alive=33.2 to 44.2 at 100M steps is a textbook symptom of naive self-play cycling.

Implementation options, from simplest to most complex:

1. **Opponent sampling from checkpoint pool**: Save prey checkpoints every M updates. When collecting predator rollouts, randomly select a prey checkpoint (uniform or prioritized). Similarly for prey training. This is the minimum viable approach.
2. **PBT over learning rates**: Run 4-8 parallel training runs. Every K updates, the worst performers copy weights from the best and perturb their LR/entropy coefficients. Fits naturally with JAX's `vmap`-over-seeds pattern.
3. **Full league training**: Multiple agent types per side with different training objectives (main agent, exploiter, etc.). Overkill for our scale.

### Implementation difficulty

**Option 1 (checkpoint pool): Low-Medium.**
- Need to save/load checkpoints during training (we already save at the end).
- Modify rollout collection to accept an opponent policy separate from the training policy.
- Main complexity: loading a random checkpoint inside a `jax.lax.scan` loop is awkward (checkpoints live on disk). Practical workaround: keep the last N opponent params in memory as a pytree stack, index with `jax.lax.dynamic_slice`.

**Option 2 (PBT): Medium.**
- Run multiple IPPO instances, periodically compare and copy/perturb.
- Can be done at the Python level (outer loop around `train_fn`) without modifying JIT-compiled code.

**Option 3 (league): High.** Not recommended at current scale.

### Expected benefit

**High for the arms race problem.** Opponent sampling should prevent the cyclical regression and stabilize the predator-prey co-learning. Based on the self-play literature, expect:
- Elimination of the arms race regression (prey_alive should plateau rather than regress).
- Possibly slower initial learning (training against a mixture of opponents is harder than training against a fixed one).
- More robust final policies that generalize across opponent strategies.

---

## 3. Communication Between Agents

### What it is

Explicit communication allows agents to send messages to each other, which are included in other agents' observations. This is distinct from implicit coordination (which emerges from shared rewards and shared weights).

Key architectures:

- **CommNet** (Sukhbaatar et al. 2016): agents broadcast continuous message vectors. At each communication round, each agent averages the messages from all other agents and feeds this into its next hidden layer. Multiple rounds allow multi-hop reasoning. Simple and differentiable.

- **TarMAC** (Das et al. 2019): targeted multi-agent communication. Agents produce messages and attention keys. Each receiving agent computes attention over all messages, attending more to relevant senders. More efficient than CommNet's mean-pooling because agents learn whose messages matter.

- **QMIX** (Rashid et al. 2018): not communication per se, but value decomposition. A mixing network combines per-agent Q-values into a joint Q-value, enforcing monotonicity (if an agent's Q-value increases, the joint Q-value increases). Designed for cooperative settings with individual rewards. Not directly applicable to our PPO setup without significant changes.

- **IC3Net** (Singh et al. 2019): agents learn a gating mechanism to decide when to communicate. Avoids flooding the communication channel.

For our predator-prey setup, communication would let predators coordinate pursuit. For example, predator A could broadcast "I'm chasing prey at position X" so other predators avoid the same target or set up an interception.

### Applicability to our setup

**Medium applicability, unclear payoff.**

Predators already share weights and observe k-nearest teammates (relative positions and velocities). This provides implicit information about what teammates are doing. The question is whether explicit messages add enough over this implicit channel.

Arguments for:
- With k_nearest_same=4 (all other predators visible), agents see teammate positions but not teammate *intentions*. A message like "I'm heading toward prey cluster at (30, 20)" could enable pincer movements.
- Communication could help solve the stacking problem by broadcasting target assignments.

Arguments against:
- With only 5 predators, the implicit coordination channel (observing all teammate positions/velocities) may be sufficient. Communication benefits scale with agent count.
- Adding communication significantly increases the action space (actions + message vector), making learning harder.
- Shared weights already provide a strong implicit coordination prior - agents with similar observations will take similar actions, and the agent index allows role differentiation.

A simpler alternative that captures most of the benefit: add a **mean-field approximation** to observations. Instead of (or in addition to) k-nearest enemies, include aggregate statistics like (count of alive prey within radius R, mean direction to nearest prey cluster). This gives agents global awareness without explicit communication.

### Implementation difficulty

**Medium-High.**

- **CommNet**: Add a communication module between hidden layers of the `ActorCritic` network. The forward pass becomes: encode obs -> broadcast/average messages -> decode with message context -> output action. Requires batching agents together within each environment (currently they're processed independently via `vmap`). Needs restructuring of how `collect_rollouts` calls the network.
- **TarMAC**: Same as CommNet but with attention instead of averaging. Slightly more parameters.
- **Mean-field obs augmentation**: **Low** difficulty. Just add a few extra features to the observation vector in `_single_agent_obs`. No architectural changes.

The main JAX challenge: communication requires agents within the same environment to interact during the forward pass. Currently, `vmap` over agents treats them independently. We'd need to restructure to process all agents in a team together, passing the full team observation tensor to a communication-aware network.

### Expected benefit

**Low-Medium.** With only 5 predators that can already see all teammates, the marginal benefit of explicit communication is likely small. The implementation cost is high relative to the expected gain.

**Recommendation**: Try mean-field observation augmentation first (add prey density/direction statistics to the observation). If that helps, consider TarMAC as a next step.

---

## 4. Curriculum Learning

### What it is

Curriculum learning starts with easier tasks and gradually increases difficulty. For predator-prey, this could mean:

1. **Prey count curriculum**: Start with fewer prey (e.g., 20) and increase to 100 as performance improves. Fewer prey means each predator needs less coordination to achieve captures, letting the basic chase behavior emerge first.

2. **Prey capability curriculum**: Start with slower, non-learning prey and gradually increase prey speed and enable prey learning. We already have `prey_speed_mult` and `prey_learn` toggles.

3. **World size curriculum**: Start with a smaller world so prey have less room to evade, then expand.

4. **Asymmetric training schedules**: Train predators for N steps against fixed prey, then train prey for M steps against fixed predators, alternating. This decouples the learning dynamics and prevents the arms race from happening continuously.

5. **Reward schedule**: Start with dense rewards (distance-based shaping) and anneal toward sparse rewards (capture-only) as agents improve. This prevents reward shaping artifacts in the final policy.

Relevant literature:
- **OpenAI Five**: used a reward shaping schedule, annealing from shaped (80% at start) to sparse rewards over training.
- **Emergent Tool Use** (Baker et al. 2019): used a simple hide-and-seek setup where emergent strategies appeared through long training with gradually increasing complexity (box manipulation tools unlocked over time).
- **Asymmetric self-play** (Sukhbaatar et al. 2017): one agent proposes tasks, another solves them. The proposer learns to create tasks just beyond the solver's current capability.

### Applicability to our setup

**High.** The infrastructure already supports most curriculum dimensions:

- `EnvConfig.n_prey` can be changed between training phases.
- `EnvConfig.prey_speed_mult` and `prey_learn` exist as toggles.
- `EnvConfig.world_size` is configurable.
- The `ScalarHParams` structure was designed for changing hyperparameters without recompilation.

The catch: changing `n_prey`, `n_predators`, or `world_size` changes observation shapes and requires recompilation (or re-initialization of the environment). Changing `prey_speed_mult` or `prey_learn` does not change shapes and can be toggled freely.

Practical curriculum plan:
1. **Phase 1** (0-10M steps): 5 predators vs 20 prey, prey non-learning, `prey_speed_mult=0.3`. Learn basic chase.
2. **Phase 2** (10-30M steps): 5 predators vs 50 prey, prey non-learning, `prey_speed_mult=0.5`. Scale up.
3. **Phase 3** (30-60M steps): 5 predators vs 100 prey, prey learning enabled, `prey_speed_mult=0.5`. Co-learning begins with predators already competent.

Problem: changing `n_prey` from 20 to 50 changes the observation size (if `k_nearest_enemy` changes) or at minimum changes the environment state shapes, requiring new model initialization or careful weight transfer. If `k_nearest_enemy` stays fixed at 3, the observation size doesn't change, and we can transfer weights directly between phases. The model only sees k-nearest enemies regardless of total prey count, so this works.

**Asymmetric training** is worth trying and easy to implement: alternate between training predators for K updates (prey policy frozen) and training prey for K updates (predator policy frozen). Modify the `_update_step` function to conditionally skip one side's PPO update based on the update counter.

### Implementation difficulty

**Low for asymmetric training.** Modify `_update_step` to conditionally apply `ppo_update` based on `update_step % (K_pred + K_prey)`. Use `jax.lax.cond` to branch.

**Medium for prey count curriculum.** Need to re-initialize the environment between phases but can preserve network weights since `k_nearest_enemy` keeps the obs size fixed. Requires breaking training into phases with separate `jax.lax.scan` calls.

**Low for speed/capability curriculum.** `prey_speed_mult` could be made a runtime parameter (passed through the carry state of `lax.scan`), avoiding recompilation.

### Expected benefit

**High for asymmetric training.** Decoupling the learning dynamics directly addresses the arms race by preventing the adversarial gradient interference that causes regression. Predators get stable targets to improve against, then prey get stable threats to evade. This is the cheapest change with the highest expected impact on the arms race problem.

**Medium for prey count curriculum.** Helps with sample efficiency (easier to learn basic coordination with fewer prey) but doesn't directly address the arms race.

---

## 5. Reward Shaping for Coordination

### What it is

The current reward structure:
- **Predators**: `capture_reward (10.0 * n_captures / n_predators)` + `distance_reward (normalized inverse distance to nearest alive prey)`. Shared capture reward, individual distance shaping.
- **Prey**: `-10.0` on capture + distance from nearest predator (higher is better).

Problems with current rewards for coordination:
1. **Predator stacking**: All predators get the same distance reward signal (approach nearest prey), so they all converge on the same target. The shared capture reward doesn't help - it rewards captures regardless of which predator did the work.
2. **No coordination incentive**: Nothing explicitly rewards spreading out or dividing territory.
3. **No pursuit persistence**: Switching targets every step is not penalized.

Reward shaping approaches:

**Anti-stacking bonus**: Penalize predators that are too close to each other. Add a negative reward proportional to the number of other predators within radius R of the agent. This directly discourages convergence.

```
anti_stack = -alpha * count(other_predators within R)
```

**Formation/spacing reward**: Reward predators for maintaining minimum inter-predator distance. A soft version: reward = `min(dist_to_nearest_predator, threshold) / threshold`. This encourages spread without requiring specific formations.

**Territory assignment**: Divide the 50x50 grid into regions (e.g., 5 Voronoi cells) and reward each predator for captures in its assigned territory. Tricky to implement without breaking differentiability, but a soft version works: reward predator i for catching prey that is closest to predator i among all predators.

**Individual capture credit**: Instead of shared `n_captures / n_predators`, give capture reward only to the predator that was closest to the captured prey. This creates individual accountability.

**Closest-predator distance reward**: Instead of rewarding each predator for its distance to nearest prey, only reward the predator that is the *uniquely closest* to each prey. Other predators get no distance reward for that prey. This naturally creates target allocation.

**Group reward for coverage**: Measure how well the predator team covers the map (e.g., sum of pairwise distances, or coverage metric). Add a small bonus for high coverage.

### Applicability to our setup

**High.** The reward functions are already extracted into `rewards.py` as pure functions. Adding new terms is straightforward:

- Anti-stacking requires computing pairwise predator distances (cheap for 5 agents).
- Individual capture credit requires knowing which predator was closest (we already compute `pred_to_prey_dists` in `_compute_rewards`).
- Coverage rewards require pairwise predator distances.

All of these are differentiable and work with JAX's `vmap`.

### Implementation difficulty

**Low.** Changes are confined to `rewards.py` and `_compute_rewards` in `predator_prey.py`. The existing architecture needs no modification.

Specific changes for "individual capture credit + anti-stacking":
1. In `compute_predator_rewards`, replace shared capture reward with per-predator credit based on which predator was closest to each captured prey.
2. Add anti-stacking penalty based on pairwise predator distances.
3. Tune `alpha` (anti-stack coefficient) and the distance threshold.

### Expected benefit

**Medium-High.** Reward shaping is the fastest path to better coordination behavior. Individual capture credit + anti-stacking should reduce the predator stacking problem within a few million steps.

Caveats:
- Reward shaping can create local optima (agents optimize the shaped reward instead of the true objective).
- Too much shaping can prevent emergent strategies.
- Anti-stacking must be balanced: too strong and predators avoid each other even when converging on a prey cluster is correct.

**Recommendation**: Start with individual capture credit (replace shared reward) and a mild anti-stacking penalty. These are low-risk, high-signal changes.

---

## 6. Attention-Based Architectures

### What it is

The current architecture uses a fixed k-nearest-neighbor observation: each agent sees the k closest same-team agents and k closest enemies. This has limitations:
- **Information loss**: Agents far away but moving toward you are invisible.
- **Fixed budget**: k=3 for enemies means the agent has no awareness of the other 97 prey.
- **No salience weighting**: The 1st and 3rd nearest enemies are treated with equal importance by the MLP.

Attention-based architectures (transformers applied to multi-agent observations) address this by processing a variable number of entities with learned importance weighting.

**Entity-based attention** (Iqbal & Sha 2019 - "Actor-Attention-Critic for Multi-Agent Reinforcement Learning"):
- Each entity (teammate, enemy) is encoded into a key-value pair.
- The agent computes attention over all entities, producing a weighted summary.
- Naturally handles variable numbers of entities (alive vs dead prey).

Architecture for our setup:
```
Input: own_state (vel, pos, id) + set of entity embeddings
  1. Embed own state: MLP -> query vector
  2. Embed each entity: MLP -> key, value vectors
  3. Multi-head attention: query attends over entity keys -> weighted sum of values
  4. Concatenate own_state_embed + attention_output -> MLP -> action_mean, value
```

**Advantages over k-nearest**:
- Can attend to all 100 prey simultaneously (or at least the top-N after a distance-based pre-filter).
- Learns which entities matter (e.g., isolated prey far from other predators might be more valuable than nearby prey in a cluster).
- Naturally handles the dead prey problem (mask out dead entities in attention).
- More parameter-efficient for large entity counts compared to flat concatenation.

**Entity types** can be encoded via type embeddings, similar to how our agent_index works but more principled. Predators and prey get different type embeddings added to their entity representations.

Relevant work:
- **MAAC** (Iqbal & Sha 2019): attention-based actor-critic for MARL.
- **Entity-based MAPPO**: Combines entity attention with MAPPO's centralized critic.
- **Waymo's motion prediction**: Uses attention over traffic agents, showing the approach scales to 100+ entities.

### Applicability to our setup

**Medium-High.** The main benefit is better observation processing, especially with 100 prey. Currently observing only k=3 nearest enemies means predators have no awareness of prey distribution. With attention, a predator could learn to head toward a dense cluster of prey far away if all nearby prey are already being chased.

However, attention over 100 entities is more expensive per forward pass than the current k=3 approach. With 5 predators and 100 prey, each predator's forward pass would attend over 104 entities (4 teammates + 100 prey). At 64 parallel envs with 256 steps per rollout, that's 64 * 5 * 256 = 81,920 forward passes per update, each attending over 104 entities. This is feasible on GPU but notably slower than the current MLP.

A practical middle ground: **attention over k-nearest** with k larger than current (e.g., k=15 enemies instead of k=3), combined with a distance-based pre-filter. This gives the attention mechanism's learned salience weighting without the cost of attending over all 100 prey.

### Implementation difficulty

**Medium-High.**

Changes required:
1. New `AttentionActorCritic` network module in `networks.py`. Replace the flat MLP with an attention-based encoder for the entity portion of the observation.
2. Change observation format: instead of a flat vector, the observation needs to be structured as (own_state, entity_list). This means modifying `_single_agent_obs` to return structured data or at least a tensor that can be reshaped into (n_entities, entity_dim).
3. The current `obs_size` function and all code that assumes flat observation vectors needs updating.
4. Masking logic for dead prey in the attention weights.

The biggest refactor is the observation structure change. Everything downstream (collector, PPO update, transitions) currently assumes `obs` is a flat `[n_agents, obs_size]` array. Moving to `[n_agents, n_entities, entity_dim]` requires changes throughout the pipeline.

A less invasive alternative: keep the flat observation but increase `k_nearest_enemy` to 10-15 and add an attention layer inside the network that reshapes the flat input back into entity form. This is architecturally less clean but requires minimal changes outside `networks.py`.

### Expected benefit

**Medium.** The benefit scales with the information gap between k-nearest and full visibility. With k=3 out of 100 prey, the agent is missing 97% of the prey information. Increasing visibility (even without full attention) should improve strategic behavior like choosing which prey cluster to approach.

For the arms race problem specifically, attention doesn't directly help. It improves the quality of individual decisions but doesn't address the co-learning instability.

**Recommendation**: Increase `k_nearest_enemy` from 3 to 8-10 first (zero implementation cost, just a config change). If that helps, consider the attention architecture.

---

## Priority Ranking

Ranked by expected-benefit-to-implementation-cost ratio for the specific problems we face (arms race regression, predator coordination):

| Priority | Approach | Addresses | Cost | Benefit |
|----------|----------|-----------|------|---------|
| 1 | **Asymmetric training** (curriculum) | Arms race | Low | High |
| 2 | **Reward shaping** (individual credit + anti-stack) | Coordination | Low | Medium-High |
| 3 | **Opponent sampling from checkpoint pool** (self-play) | Arms race | Low-Medium | High |
| 4 | **Increase k_nearest_enemy** (config change) | Information | None | Low-Medium |
| 5 | **MAPPO centralized critic** | Coordination | Medium | Medium-High |
| 6 | **Prey count curriculum** | Sample efficiency | Medium | Medium |
| 7 | **Mean-field obs augmentation** | Information | Low | Low-Medium |
| 8 | **Attention architecture** | Information | Medium-High | Medium |
| 9 | **PBT for hyperparameters** | Tuning | Medium | Medium |
| 10 | **Explicit communication** (CommNet/TarMAC) | Coordination | Medium-High | Low-Medium |

Items 1-3 directly address the two main problems (arms race and coordination) with reasonable implementation cost. Item 4 is free and should be tried immediately. Items 5-6 are solid medium-term improvements. Items 7-10 are lower priority given current bottlenecks.
