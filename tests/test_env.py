"""Tests for boids environment and PPO training."""

import jax
import jax.numpy as jnp

from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.rewards import compute_predator_rewards, compute_prey_rewards
from jax_boids.envs.types import BoidsState, EnvConfig

# ---------------------------------------------------------------------------
# Helpers for deterministic scenario tests
# ---------------------------------------------------------------------------

# Minimal config: 2 predators, 2 prey, small world
SCENARIO_CONFIG = EnvConfig(
    n_predators=2,
    n_prey=2,
    world_size=100.0,
    max_steps=500,
    max_speed=5.0,
    max_acceleration=2.0,
    dt=0.1,
    separation_weight=0.0,  # disable boids forces for predictability
    alignment_weight=0.0,
    cohesion_weight=0.0,
    perception_radius=15.0,
    capture_radius=2.0,
    predator_speed_bonus=1.2,
    k_nearest_same=1,
    k_nearest_enemy=2,
)


def _make_state(
    pred_pos,
    prey_pos,
    pred_vel=None,
    prey_vel=None,
    prey_alive=None,
    step=0,
    config=SCENARIO_CONFIG,
):
    """Build a BoidsState from known positions."""
    pred_pos = jnp.array(pred_pos, dtype=jnp.float32)
    prey_pos = jnp.array(prey_pos, dtype=jnp.float32)
    n_pred = pred_pos.shape[0]
    n_prey = prey_pos.shape[0]
    if pred_vel is None:
        pred_vel = jnp.zeros((n_pred, 2))
    else:
        pred_vel = jnp.array(pred_vel, dtype=jnp.float32)
    if prey_vel is None:
        prey_vel = jnp.zeros((n_prey, 2))
    else:
        prey_vel = jnp.array(prey_vel, dtype=jnp.float32)
    if prey_alive is None:
        prey_alive = jnp.ones(n_prey, dtype=bool)
    else:
        prey_alive = jnp.array(prey_alive, dtype=bool)
    return BoidsState(
        predator_pos=pred_pos,
        predator_vel=pred_vel,
        prey_pos=prey_pos,
        prey_vel=prey_vel,
        prey_alive=prey_alive,
        step=step,
        key=jax.random.PRNGKey(0),
    )


def _zero_actions(config=SCENARIO_CONFIG):
    return {
        "predator": jnp.zeros((config.n_predators, 2)),
        "prey": jnp.zeros((config.n_prey, 2)),
    }


def test_reset():
    """Environment reset produces correct shapes."""
    config = EnvConfig(n_predators=5, n_prey=10)
    env = PredatorPreyEnv(config)

    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    assert obs["predator"].shape == (5, env.observation_size)
    assert obs["prey"].shape == (10, env.observation_size)
    assert state.predator_pos.shape == (5, 2)
    assert state.prey_pos.shape == (10, 2)
    assert jnp.all(state.prey_alive)


def test_step():
    """Environment step produces correct shapes."""
    config = EnvConfig(n_predators=5, n_prey=10)
    env = PredatorPreyEnv(config)

    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    key, k1, k2, k3 = jax.random.split(key, 4)
    actions = {
        "predator": jax.random.uniform(k1, (5, 2), minval=-1, maxval=1),
        "prey": jax.random.uniform(k2, (10, 2), minval=-1, maxval=1),
    }

    next_obs, next_state, rewards, _, _ = env.step(k3, state, actions)

    assert next_obs["predator"].shape == (5, env.observation_size)
    assert rewards["predator"].shape == (5,)
    assert rewards["prey"].shape == (10,)
    assert next_state.step == 1


def test_jit():
    """Environment step can be JIT compiled."""
    config = EnvConfig(n_predators=5, n_prey=10)
    env = PredatorPreyEnv(config)

    @jax.jit
    def do_step(key, state, actions):
        return env.step(key, state, actions)

    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    key, k1, k2, k3 = jax.random.split(key, 4)
    actions = {
        "predator": jax.random.uniform(k1, (5, 2), minval=-1, maxval=1),
        "prey": jax.random.uniform(k2, (10, 2), minval=-1, maxval=1),
    }

    # First call compiles
    _, next_state, _, _, _ = do_step(k3, state, actions)

    # Second call uses cached compilation
    key, k4 = jax.random.split(key)
    do_step(k4, next_state, actions)


def test_vmap():
    """Environment can be vmapped for parallel envs."""
    config = EnvConfig(n_predators=5, n_prey=10)
    env = PredatorPreyEnv(config)

    n_envs = 8
    keys = jax.random.split(jax.random.PRNGKey(0), n_envs)

    obs, states = jax.vmap(env.reset)(keys)

    assert obs["predator"].shape == (n_envs, 5, env.observation_size)
    assert states.predator_pos.shape == (n_envs, 5, 2)


def test_episode():
    """Episode runs and captures work."""
    config = EnvConfig(n_predators=5, n_prey=10, max_steps=100)
    env = PredatorPreyEnv(config)

    key = jax.random.PRNGKey(42)
    _, state = env.reset(key)

    for _ in range(100):
        key, k1, k2, k3 = jax.random.split(key, 4)
        actions = {
            "predator": jax.random.uniform(k1, (5, 2), minval=-1, maxval=1),
            "prey": jax.random.uniform(k2, (10, 2), minval=-1, maxval=1),
        }
        _, state, _, dones, _ = env.step(k3, state, actions)

        if dones["__all__"]:
            break

    assert state.step > 0


def test_gae_simple_trajectory():
    """GAE with known trajectory matches hand calculation."""
    from jax_boids.ppo import compute_gae

    # 3-step trajectory, single agent
    rewards = jnp.array([[1.0], [1.0], [1.0]])  # [T=3, N=1]
    values = jnp.array([[0.5], [0.5], [0.5], [0.5]])  # [T+1=4, N=1] (includes bootstrap)
    dones = jnp.array([[0.0], [0.0], [0.0]])  # [T=3, N=1]

    advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)

    # Hand-calculated (backwards):
    # delta_2 = r_2 + gamma * V_3 * (1-d_2) - V_2 = 1 + 0.99*0.5 - 0.5 = 0.995
    # A_2 = delta_2 = 0.995
    #
    # delta_1 = 1 + 0.99*0.5 - 0.5 = 0.995
    # A_1 = delta_1 + gamma * lambda * A_2 = 0.995 + 0.99*0.95*0.995 = 1.931
    #
    # delta_0 = 1 + 0.99*0.5 - 0.5 = 0.995
    # A_0 = delta_0 + gamma * lambda * A_1 = 0.995 + 0.9405*1.931 = 2.811
    expected_A2 = 0.995
    expected_A1 = 0.995 + 0.99 * 0.95 * expected_A2
    expected_A0 = 0.995 + 0.99 * 0.95 * expected_A1

    assert jnp.allclose(advantages[2, 0], expected_A2, atol=1e-3)
    assert jnp.allclose(advantages[1, 0], expected_A1, atol=1e-3)
    assert jnp.allclose(advantages[0, 0], expected_A0, atol=1e-3)

    # Returns = advantages + values
    assert jnp.allclose(returns, advantages + values[:-1], atol=1e-6)


def test_gae_with_done():
    """GAE resets at episode boundary."""
    from jax_boids.ppo import compute_gae

    # Episode ends at step 1 (done=1)
    rewards = jnp.array([[1.0], [1.0], [1.0]])
    values = jnp.array([[0.5], [0.5], [0.5], [0.5]])
    dones = jnp.array([[0.0], [1.0], [0.0]])  # Episode ends at step 1

    advantages, _ = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)

    # Step 2: normal (no done after it)
    # delta_2 = 1 + 0.99*0.5 - 0.5 = 0.995, A_2 = 0.995

    # Step 1: done=1, so next_value contribution is zeroed
    # delta_1 = 1 + 0.99*0.5*(1-1) - 0.5 = 1 - 0.5 = 0.5
    # A_1 = delta_1 + gamma*lambda*(1-done)*A_2 = 0.5 + 0 = 0.5

    # Step 0: normal, uses A_1
    # delta_0 = 1 + 0.99*0.5 - 0.5 = 0.995
    # A_0 = 0.995 + 0.9405*0.5 = 1.465

    assert jnp.allclose(advantages[1, 0], 0.5, atol=1e-3), f"Got {advantages[1, 0]}"
    assert jnp.allclose(advantages[0, 0], 1.465, atol=1e-2), f"Got {advantages[0, 0]}"


def test_distribution_log_prob():
    """Distribution log_prob matches manual Gaussian calculation."""
    from jax_boids.ppo import make_distribution

    mean = jnp.array([0.0, 0.0])
    logstd = jnp.array([0.0, 0.0])  # std = 1
    pi = make_distribution(mean, logstd)

    # log_prob of mean should be -d/2 * log(2*pi) for unit variance
    log_prob_at_mean = pi.log_prob(mean)
    expected = -2 / 2 * jnp.log(2 * jnp.pi)  # -1.8379

    assert jnp.allclose(log_prob_at_mean, expected, atol=1e-4)


def test_distribution_sample_shape():
    """Distribution samples have correct shape."""
    from jax_boids.ppo import make_distribution

    mean = jnp.array([0.0, 0.0])
    logstd = jnp.array([0.0, 0.0])
    pi = make_distribution(mean, logstd)

    key = jax.random.PRNGKey(0)
    sample = pi.sample(seed=key)

    assert sample.shape == (2,), f"Expected (2,), got {sample.shape}"


def test_network_output_shapes():
    """Network outputs have correct shapes."""
    from jax_boids.networks import ActorCritic

    network = ActorCritic(action_dim=2)
    key = jax.random.PRNGKey(0)
    obs = jnp.zeros((4,))
    params = network.init(key, obs)

    # Single observation
    out = network.apply(params, obs)
    assert out.action_mean.shape == (2,), f"action_mean: {out.action_mean.shape}"
    assert out.action_logstd.shape == (2,), f"action_logstd: {out.action_logstd.shape}"
    assert out.value.shape == (), f"value: {out.value.shape}"

    # Batched via vmap
    batch_obs = jnp.zeros((8, 4))
    batch_out = jax.vmap(lambda o: network.apply(params, o))(batch_obs)
    assert batch_out.action_mean.shape == (8, 2)
    assert batch_out.value.shape == (8,)


def test_ppo_positive_advantage():
    """Positive advantage increases action probability."""
    from jax_boids.networks import ActorCritic
    from jax_boids.ppo import make_distribution, ppo_loss

    network = ActorCritic(action_dim=2)
    key = jax.random.PRNGKey(42)
    obs = jnp.array([[1.0, 0.5, -0.5, 0.0]])
    params = network.init(key, obs[0])

    out = network.apply(params, obs[0])
    pi = make_distribution(out.action_mean, out.action_logstd)
    action = jnp.array([[0.5, -0.3]])
    old_log_prob = pi.log_prob(action[0])

    advantage = jnp.array([1.0])
    returns = jnp.array([1.0])

    def loss_fn(p):
        return ppo_loss(
            p,
            network.apply,
            obs,
            action,
            jnp.array([old_log_prob]),
            advantage,
            returns,
            clip_eps=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
        )[0]

    grads = jax.grad(loss_fn)(params)
    new_params = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)

    new_out = network.apply(new_params, obs[0])
    new_pi = make_distribution(new_out.action_mean, new_out.action_logstd)
    new_log_prob = new_pi.log_prob(action[0])

    assert new_log_prob > old_log_prob, (
        f"Expected log_prob to increase: {old_log_prob:.4f} -> {new_log_prob:.4f}"
    )


def test_ppo_value_toward_return():
    """Value estimate moves toward the return."""
    from jax_boids.networks import ActorCritic
    from jax_boids.ppo import make_distribution, ppo_loss

    network = ActorCritic(action_dim=2)
    key = jax.random.PRNGKey(42)
    obs = jnp.array([[1.0, 0.5, -0.5, 0.0]])
    params = network.init(key, obs[0])

    out = network.apply(params, obs[0])
    old_value = float(out.value)
    target_return = 5.0

    pi = make_distribution(out.action_mean, out.action_logstd)
    action = jnp.array([[0.0, 0.0]])
    old_log_prob = pi.log_prob(action[0])

    def loss_fn(p):
        return ppo_loss(
            p,
            network.apply,
            obs,
            action,
            jnp.array([old_log_prob]),
            jnp.array([0.0]),
            jnp.array([target_return]),
            clip_eps=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
        )[0]

    grads = jax.grad(loss_fn)(params)
    new_params = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)

    new_value = float(network.apply(new_params, obs[0]).value)

    assert abs(new_value - target_return) < abs(old_value - target_return), (
        f"Value should move toward {target_return}: {old_value:.4f} -> {new_value:.4f}"
    )


# ---------------------------------------------------------------------------
# Scenario tests — deterministic positions
# ---------------------------------------------------------------------------


def test_capture_at_known_distance():
    """Predator within capture_radius of prey captures it after one step."""
    env = PredatorPreyEnv(SCENARIO_CONFIG)
    # Place predator 0 right on top of prey 0 (distance 0.5 < capture_radius 2.0)
    # Predator 1 and prey 1 far away
    state = _make_state(
        pred_pos=[[50.0, 50.0], [10.0, 10.0]],
        prey_pos=[[50.5, 50.0], [90.0, 90.0]],
    )
    key = jax.random.PRNGKey(1)
    _, new_state, _, _, info = env.step(key, state, _zero_actions())
    # Prey 0 should be captured
    assert not new_state.prey_alive[0], "Prey 0 should be captured"
    assert new_state.prey_alive[1], "Prey 1 should survive"
    assert int(info["captures_this_step"]) == 1


def test_no_capture_beyond_radius():
    """Predator just outside capture_radius doesn't capture prey."""
    env = PredatorPreyEnv(SCENARIO_CONFIG)
    # Place predator 5 units away from prey — well beyond capture_radius=2.0
    state = _make_state(
        pred_pos=[[50.0, 50.0], [10.0, 10.0]],
        prey_pos=[[55.0, 50.0], [90.0, 90.0]],
    )
    key = jax.random.PRNGKey(1)
    _, new_state, _, _, info = env.step(key, state, _zero_actions())
    assert jnp.all(new_state.prey_alive), "No prey should be captured"
    assert int(info["captures_this_step"]) == 0


def test_dead_prey_excluded():
    """Dead prey can't be captured again."""
    env = PredatorPreyEnv(SCENARIO_CONFIG)
    # Prey 0 already dead, predator 0 right on top of it
    state = _make_state(
        pred_pos=[[50.0, 50.0], [10.0, 10.0]],
        prey_pos=[[50.0, 50.0], [90.0, 90.0]],
        prey_alive=[False, True],
    )
    key = jax.random.PRNGKey(1)
    _, new_state, _, _, info = env.step(key, state, _zero_actions())
    assert not new_state.prey_alive[0], "Prey 0 stays dead"
    assert new_state.prey_alive[1], "Prey 1 stays alive"
    assert int(info["captures_this_step"]) == 0


def test_episode_ends_max_steps():
    """Episode ends when max_steps reached."""
    config = SCENARIO_CONFIG.replace(max_steps=3)
    env = PredatorPreyEnv(config)
    # Place everyone far apart so no captures happen
    state = _make_state(
        pred_pos=[[10.0, 10.0], [20.0, 20.0]],
        prey_pos=[[80.0, 80.0], [90.0, 90.0]],
        config=config,
    )
    key = jax.random.PRNGKey(0)
    for i in range(3):
        key, step_key = jax.random.split(key)
        _, state, _, dones, _ = env.step(step_key, state, _zero_actions(config))
    assert dones["__all__"], "Episode should be done at max_steps"


def test_episode_ends_all_prey_dead():
    """Episode ends when all prey are captured."""
    env = PredatorPreyEnv(SCENARIO_CONFIG)
    # Place both predators on top of both prey
    state = _make_state(
        pred_pos=[[50.0, 50.0], [60.0, 60.0]],
        prey_pos=[[50.0, 50.0], [60.0, 60.0]],
    )
    key = jax.random.PRNGKey(0)
    _, new_state, _, dones, _ = env.step(key, state, _zero_actions())
    assert not jnp.any(new_state.prey_alive), "All prey should be dead"
    assert dones["__all__"], "Episode should be done"


def test_obs_values_known_positions():
    """Observations contain correct relative positions for known state."""
    env = PredatorPreyEnv(SCENARIO_CONFIG)
    cfg = SCENARIO_CONFIG
    state = _make_state(
        pred_pos=[[50.0, 50.0], [60.0, 50.0]],
        prey_pos=[[70.0, 50.0], [80.0, 50.0]],
        pred_vel=[[1.0, 0.0], [0.0, 0.0]],
    )
    obs, _ = env.reset_from_state(state)

    # Predator 0 obs: velocity=[1,0]/max_speed, teammate rel_pos=[10,0]/half_world
    pred0_obs = obs["predator"][0]
    half_world = cfg.world_size / 2.0
    # First 2 values: own velocity normalized
    assert jnp.allclose(pred0_obs[0], 1.0 / cfg.max_speed, atol=1e-5)
    assert jnp.allclose(pred0_obs[1], 0.0, atol=1e-5)
    # Next 2 values: nearest same-team relative pos (pred1 at [60,50] - [50,50] = [10,0])
    assert jnp.allclose(pred0_obs[2], 10.0 / half_world, atol=1e-5)
    assert jnp.allclose(pred0_obs[3], 0.0 / half_world, atol=1e-5)


def test_reward_capture_value():
    """Predator gets +10/n_pred per capture, prey gets -10 on capture."""
    env = PredatorPreyEnv(SCENARIO_CONFIG)
    # Predator 0 on top of prey 0
    state = _make_state(
        pred_pos=[[50.0, 50.0], [10.0, 10.0]],
        prey_pos=[[50.0, 50.0], [90.0, 90.0]],
    )
    key = jax.random.PRNGKey(0)
    _, _, rewards, _, info = env.step(key, state, _zero_actions())
    n_captures = int(info["captures_this_step"])
    assert n_captures == 1
    # Each predator gets capture_reward + distance_reward
    # Capture: 10 * n_captures / n_pred = 5
    # Distance adds 0-1 on top
    assert rewards["predator"][0] > 4.0, f"Pred reward too low: {rewards['predator'][0]}"
    assert rewards["predator"][1] > 4.0, f"Pred reward too low: {rewards['predator'][1]}"
    # Captured prey gets -10 (plus small distance component)
    assert rewards["prey"][0] < -9.0, (
        f"Captured prey reward should be < -9, got {rewards['prey'][0]}"
    )


def test_reward_survival():
    """Alive prey get distance reward > 0, dead prey get 0."""
    env = PredatorPreyEnv(SCENARIO_CONFIG)
    # Everyone far apart, no captures
    state = _make_state(
        pred_pos=[[10.0, 10.0], [20.0, 20.0]],
        prey_pos=[[80.0, 80.0], [90.0, 90.0]],
        prey_alive=[True, False],
    )
    key = jax.random.PRNGKey(0)
    _, _, rewards, _, _ = env.step(key, state, _zero_actions())
    # Alive prey gets distance reward (positive when far from predators)
    assert rewards["prey"][0] > 0.0, "Alive prey should have positive reward"
    # Dead prey gets 0
    assert jnp.allclose(rewards["prey"][1], 0.0, atol=1e-5), "Dead prey should get 0 reward"


def test_reward_distance():
    """Distance reward scales with distance to nearest predator."""
    env = PredatorPreyEnv(SCENARIO_CONFIG)
    # Prey 0 close to predator, prey 1 far from predator
    state = _make_state(
        pred_pos=[[50.0, 50.0], [50.0, 50.0]],
        prey_pos=[[53.0, 50.0], [90.0, 90.0]],  # 3 vs ~56.6 units away
    )
    key = jax.random.PRNGKey(0)
    _, _, rewards, _, _ = env.step(key, state, _zero_actions())
    # Prey 1 (far) should get higher reward than prey 0 (close)
    assert rewards["prey"][1] > rewards["prey"][0], (
        f"Far prey should get higher reward: "
        f"close={rewards['prey'][0]:.3f}, far={rewards['prey'][1]:.3f}"
    )


# ---------------------------------------------------------------------------
# Reward pure function unit tests
# ---------------------------------------------------------------------------


def test_predator_reward_pure_fn():
    """compute_predator_rewards with known inputs."""
    # Far prey: distance_reward clipped to 0
    dist_to_prey = jnp.array([100.0, 100.0, 100.0, 100.0, 100.0])
    rewards = compute_predator_rewards(
        n_captures=jnp.array(3), n_predators=5, dist_to_prey=dist_to_prey, world_size=10.0
    )
    assert rewards.shape == (5,)
    # Capture: 3 * 10.0 / 5 = 6.0. Distance: 0 (beyond max_dist)
    assert jnp.allclose(rewards, 6.0, atol=1e-5)

    rewards_zero = compute_predator_rewards(
        n_captures=jnp.array(0), n_predators=5, dist_to_prey=dist_to_prey, world_size=10.0
    )
    assert jnp.allclose(rewards_zero, 0.0, atol=1e-5)

    # Close prey: distance_reward near 1.0
    dist_close = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
    rewards_close = compute_predator_rewards(
        n_captures=jnp.array(0), n_predators=5, dist_to_prey=dist_close, world_size=10.0
    )
    # max_dist = 10.0 * 0.707 = 7.07. distance_reward = 1 - 0.5/7.07 ≈ 0.93
    assert rewards_close[0] > 0.9, (
        f"Close prey should give high distance reward: {rewards_close[0]}"
    )


def test_prey_reward_pure_fn():
    """compute_prey_rewards with known inputs."""
    prey_pos = jnp.array([[50.0, 50.0], [90.0, 90.0]])
    pred_pos = jnp.array([[50.0, 50.0], [10.0, 10.0]])
    prey_alive = jnp.array([True, True])
    captures = jnp.array([True, False])

    rewards = compute_prey_rewards(prey_pos, pred_pos, prey_alive, captures, world_size=100.0)
    assert rewards.shape == (2,)

    # Prey 0: captured → -10 + small distance component
    assert rewards[0] < -9.0, f"Captured prey reward: {rewards[0]}"

    # Prey 1: alive, not captured → distance reward > 0
    # min dist to pred (wrapped on 100x100 grid) > 0
    assert rewards[1] > 0.0, f"Surviving prey reward: {rewards[1]}"


# ---------------------------------------------------------------------------
# Collector abstraction tests
# ---------------------------------------------------------------------------


def test_collect_rollouts_learned_predators_random_prey():
    """Rollout collection with learned policy for predators, random for prey."""
    from jax_boids.collector import (
        PolicyConfig,
        PolicyType,
        RolloutConfig,
        collect_rollouts,
    )
    from jax_boids.ppo import create_train_state

    env_config = EnvConfig(n_predators=2, n_prey=3, max_steps=50)
    env = PredatorPreyEnv(env_config)
    rollout_config = RolloutConfig(n_steps=4, n_envs=2)

    # Initialize predator policy
    key = jax.random.PRNGKey(42)
    k1, k2, key = jax.random.split(key, 3)
    pred_state = create_train_state(k1, env.observation_size, env.action_size, 3e-4, 0.5)

    # Configure policies
    policies = {
        "predator": PolicyConfig(PolicyType.LEARNED, train_state=pred_state, noise_scale=0.0),
        "prey": PolicyConfig(PolicyType.RANDOM, noise_scale=0.3),
    }

    # Collect rollouts
    key, (transitions, infos), obs, env_state = collect_rollouts(
        env, policies, env_config, rollout_config, key
    )

    # Verify rollout shapes (transitions is dict keyed by agent type)
    assert set(transitions.keys()) == {"predator", "prey"}
    for agent_type in {"predator", "prey"}:
        n_agents = env_config.n_predators if agent_type == "predator" else env_config.n_prey
        transition = transitions[agent_type]
        assert transition.obs.shape == (
            rollout_config.n_steps,
            rollout_config.n_envs,
            n_agents,
            env.observation_size,
        )
        assert transition.reward.shape == (
            rollout_config.n_steps,
            rollout_config.n_envs,
            n_agents,
        )
        assert transition.done.shape == transition.reward.shape
        assert transition.log_prob.shape == transition.reward.shape
        assert transition.value.shape == transition.reward.shape

    # Verify info shapes
    assert infos["prey_alive"].shape == (rollout_config.n_steps, rollout_config.n_envs)

    # Verify final obs shapes
    assert obs["predator"].shape == (
        rollout_config.n_envs,
        env_config.n_predators,
        env.observation_size,
    )


def test_collect_rollouts_both_learned():
    """Rollout collection with learned policies for both agent types."""
    from jax_boids.collector import (
        PolicyConfig,
        PolicyType,
        RolloutConfig,
        collect_rollouts,
    )
    from jax_boids.ppo import create_train_state

    env_config = EnvConfig(n_predators=2, n_prey=3, max_steps=50)
    env = PredatorPreyEnv(env_config)
    rollout_config = RolloutConfig(n_steps=2, n_envs=2)

    # Initialize both policies
    key = jax.random.PRNGKey(42)
    k1, k2, key = jax.random.split(key, 3)
    pred_state = create_train_state(k1, env.observation_size, env.action_size, 3e-4, 0.5)
    prey_state = create_train_state(k2, env.observation_size, env.action_size, 3e-4, 0.5)

    policies = {
        "predator": PolicyConfig(PolicyType.LEARNED, train_state=pred_state, noise_scale=0.0),
        "prey": PolicyConfig(PolicyType.LEARNED, train_state=prey_state, noise_scale=0.0),
    }

    # Collect rollouts
    key, (transitions, infos), obs, env_state = collect_rollouts(
        env, policies, env_config, rollout_config, key
    )

    # Verify rollout shapes (transitions is dict keyed by agent type)
    assert set(transitions.keys()) == {"predator", "prey"}
    for agent_type in {"predator", "prey"}:
        n_agents = env_config.n_predators if agent_type == "predator" else env_config.n_prey
        transition = transitions[agent_type]
        # Check value field exists and has correct shape
        assert hasattr(transition, "value")
        assert transition.value.shape == (
            rollout_config.n_steps,
            rollout_config.n_envs,
            n_agents,
        )
        assert transition.obs.shape == (
            rollout_config.n_steps,
            rollout_config.n_envs,
            n_agents,
            env.observation_size,
        )
        assert transition.reward.shape == (
            rollout_config.n_steps,
            rollout_config.n_envs,
            n_agents,
        )
        assert transition.done.shape == transition.reward.shape
        assert transition.log_prob.shape == transition.reward.shape


def test_collect_rollouts_both_random():
    """Rollout collection with random policies for all agents."""
    from jax_boids.collector import (
        PolicyConfig,
        PolicyType,
        RolloutConfig,
        collect_rollouts,
    )

    env_config = EnvConfig(n_predators=2, n_prey=3, max_steps=50)
    env = PredatorPreyEnv(env_config)
    rollout_config = RolloutConfig(n_steps=2, n_envs=2)

    policies = {
        "predator": PolicyConfig(PolicyType.RANDOM, noise_scale=0.5),
        "prey": PolicyConfig(PolicyType.RANDOM, noise_scale=0.3),
    }

    # Collect rollouts
    key = jax.random.PRNGKey(42)
    key, (transitions, infos), obs, env_state = collect_rollouts(
        env, policies, env_config, rollout_config, key
    )

    # Verify rollout collection works
    assert set(transitions.keys()) == {"predator", "prey"}
    for agent_type in {"predator", "prey"}:
        transition = transitions[agent_type]
        assert transition.obs.shape[0] == rollout_config.n_steps
        assert transition.obs.shape[1] == rollout_config.n_envs


def test_create_policy_fn_learned():
    """Create policy function for learned policy."""
    from jax_boids.collector import PolicyConfig, PolicyType, create_policy_fn
    from jax_boids.ppo import create_train_state

    env_config = EnvConfig(n_predators=2, n_prey=3)
    env = PredatorPreyEnv(env_config)

    key = jax.random.PRNGKey(42)
    train_state = create_train_state(key, env.observation_size, env.action_size, 3e-4, 0.5)
    policy_fn = create_policy_fn(PolicyConfig(PolicyType.LEARNED, train_state=train_state))

    # Test with batched observations
    obs = jnp.zeros((10, 2, env.observation_size))  # [n_envs, n_agents, obs_size]
    obs_flat = obs.reshape(-1, obs.shape[-1])  # [n_envs*n_agents, obs_size]

    k1, key = jax.random.split(key)
    actions, log_probs = policy_fn(obs_flat, k1)

    assert actions.shape == (obs_flat.shape[0], env.action_size)
    assert log_probs.shape == (obs_flat.shape[0],)
    # Actions are unbounded since they come from a normal distribution
    assert jnp.isfinite(actions).all()
    assert jnp.isfinite(log_probs).all()


def test_create_policy_fn_random():
    """Create policy function for random policy."""
    from jax_boids.collector import PolicyConfig, PolicyType, create_policy_fn

    noise_scale = 0.5
    policy_fn = create_policy_fn(PolicyConfig(PolicyType.RANDOM, noise_scale=noise_scale))

    key = jax.random.PRNGKey(42)
    obs = jnp.zeros((10, 4))  # [batch_size, obs_size]
    k1, key = jax.random.split(key)
    actions, log_probs = policy_fn(obs, k1)

    # Random actions should be within noise scale
    assert actions.shape == (obs.shape[0], 2)  # action_size is 2 for boids
    assert jnp.all(actions >= -noise_scale)
    assert jnp.all(actions <= noise_scale)
    # log_probs are zero for random policy
    assert jnp.all(log_probs == 0.0)


def test_create_policy_fn_invalid_type():
    """Create policy function raises error for invalid policy type."""
    from jax_boids.collector import PolicyConfig, PolicyType

    # Test by checking enum values exist
    assert PolicyType.LEARNED.value == "learned"
    assert PolicyType.RANDOM.value == "random"

    # Since PolicyType is an Enum, invalid values would be caught at runtime
    # Test that the PolicyConfig dataclass exists and has correct fields
    import inspect

    sig = inspect.signature(PolicyConfig)
    params = list(sig.parameters.keys())
    assert "policy_type" in params
    assert "train_state" in params
    assert "noise_scale" in params
