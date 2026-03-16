"""Diagnostic: verify learning signal and action application for both sides."""

import jax
import jax.numpy as jnp

from jax_boids.collector import PolicyConfig, PolicyType, RolloutConfig, collect_rollouts
from jax_boids.envs.predator_prey import PredatorPreyEnv
from jax_boids.envs.types import EnvConfig
from jax_boids.ppo import create_train_state, ppo_update


def main():
    env_config = EnvConfig(
        n_predators=2,
        n_prey=5,
        prey_learn=True,
        boids_strength=0.3,
        predator_speed_bonus=1.5,
        prey_speed_mult=0.5,
    )
    env = PredatorPreyEnv(env_config)
    key = jax.random.PRNGKey(0)

    print("=" * 60)
    print("1. REWARD SIGNAL CHECK")
    print("=" * 60)

    # Run 50 steps with zero actions
    k1, key = jax.random.split(key)
    obs, state = env.reset(k1)
    zero_pred = jnp.zeros((2, 2))
    zero_prey = jnp.zeros((5, 2))

    pred_rewards_zero = []
    prey_rewards_zero = []
    for i in range(50):
        k1, key = jax.random.split(key)
        obs, state, rewards, dones, info = env.step(
            k1, state, {"predator": zero_pred, "prey": zero_prey}
        )
        pred_rewards_zero.append(rewards["predator"])
        prey_rewards_zero.append(rewards["prey"])

    pred_r = jnp.stack(pred_rewards_zero)
    prey_r = jnp.stack(prey_rewards_zero)
    print("Zero actions (50 steps):")
    print(
        f"  Pred reward per step: mean={pred_r.mean():.4f} std={pred_r.std():.4f} "
        f"min={pred_r.min():.4f} max={pred_r.max():.4f}"
    )
    print(
        f"  Prey reward per step: mean={prey_r.mean():.4f} std={prey_r.std():.4f} "
        f"min={prey_r.min():.4f} max={prey_r.max():.4f}"
    )

    # Run 50 steps with max predator actions toward prey
    k1, key = jax.random.split(key)
    obs, state = env.reset(k1)
    pred_rewards_chase = []
    prey_rewards_chase = []
    for i in range(50):
        k1, key = jax.random.split(key)
        # Predators move toward nearest prey (use obs enemy_rel_pos)
        chase_pred = jnp.ones((2, 2))  # Max action
        flee_prey = -jnp.ones((5, 2))  # Max opposite action
        obs, state, rewards, dones, info = env.step(
            k1, state, {"predator": chase_pred, "prey": flee_prey}
        )
        pred_rewards_chase.append(rewards["predator"])
        prey_rewards_chase.append(rewards["prey"])

    pred_rc = jnp.stack(pred_rewards_chase)
    prey_rc = jnp.stack(prey_rewards_chase)
    print("\nMax actions (50 steps):")
    print(f"  Pred reward per step: mean={pred_rc.mean():.4f} std={pred_rc.std():.4f}")
    print(f"  Prey reward per step: mean={prey_rc.mean():.4f} std={prey_rc.std():.4f}")
    print(
        f"  Reward difference (max vs zero): pred={pred_rc.mean() - pred_r.mean():.4f} "
        f"prey={prey_rc.mean() - prey_r.mean():.4f}"
    )

    print("\n" + "=" * 60)
    print("2. DISTANCE REWARD SANITY CHECK")
    print("=" * 60)
    # On 10x10 grid, max wrapped distance is sqrt(5^2+5^2)=7.07
    for dist in [0.0, 1.0, 3.0, 5.0, 7.07]:
        r = (50.0 - dist) * 0.05
        print(f"  dist={dist:.1f} → pred_distance_reward={r:.3f}")
    print("  ^^ These should vary a LOT but they only range 2.15-2.50!")
    print("  The 50.0 threshold was designed for a bigger world.")

    print("\n" + "=" * 60)
    print("3. PREY ACTION vs BOIDS FORCE MAGNITUDE")
    print("=" * 60)
    k1, key = jax.random.split(key)
    obs, state = env.reset(k1)

    from jax_boids.envs.boids import compute_boids_forces

    boids_force = compute_boids_forces(
        state.prey_pos,
        state.prey_vel,
        env_config.perception_radius,
        env_config.separation_weight * 0.3,
        env_config.alignment_weight * 0.3,
        env_config.cohesion_weight * 0.3,
        env_config.world_size,
    )
    action_force = jnp.ones((5, 2)) * env_config.max_acceleration  # max learned action

    boids_mag = jnp.linalg.norm(boids_force, axis=-1)
    action_mag = jnp.linalg.norm(action_force, axis=-1)
    print(f"  Boids force magnitude per prey: {boids_mag}")
    print(f"  Boids mean: {boids_mag.mean():.3f}")
    print(f"  Max action force magnitude: {action_mag[0]:.3f}")
    print(f"  Ratio (boids/action): {boids_mag.mean() / action_mag[0]:.2f}x")
    if boids_mag.mean() > action_mag[0]:
        print("  !! BOIDS DOMINATE ACTIONS — prey can barely steer !!")

    print("\n" + "=" * 60)
    print("4. OBSERVATION CHECK")
    print("=" * 60)
    ks = env_config.k_nearest_same
    ke = env_config.k_nearest_enemy
    np_ = env_config.n_predators
    ny = env_config.n_prey
    print(f"  Obs size: {env.observation_size}")
    print(f"  k_nearest_same={ks}, k_nearest_enemy={ke}")
    pred_see = min(ks, np_ - 1)
    prey_see = min(ks, ny - 1)
    print(f"  Predators: {np_} -> see {pred_see} same, pad to {ks} ({ks - pred_see} zeros)")
    print(f"  Prey: {ny} -> see {prey_see} same, pad to {ks} ({max(0, ks - prey_see)} zeros)")
    print(f"  Pred enemy: {min(ke, ny)}/{ke} slots")
    print(f"  Prey enemy: {min(ke, np_)}/{ke} slots ({ke - min(ke, np_)} zero)")

    pred_obs = obs["predator"][0]  # first predator
    prey_obs = obs["prey"][0]  # first prey
    print(f"\n  Pred obs sample: {pred_obs}")
    print(f"  Prey obs sample: {prey_obs}")
    n_zero_pred = (pred_obs == 0).sum()
    n_zero_prey = (prey_obs == 0).sum()
    print(f"  Pred zero dims: {n_zero_pred}/{env.observation_size}")
    print(f"  Prey zero dims: {n_zero_prey}/{env.observation_size}")

    print("\n" + "=" * 60)
    print("5. PPO GRADIENT FLOW CHECK")
    print("=" * 60)

    # Create networks and run one PPO update
    k1, k2, key = jax.random.split(key, 3)
    pred_state = create_train_state(
        k1, env.observation_size, env.action_size, lr=2.7e-3, max_grad_norm=0.5
    )
    prey_state = create_train_state(
        k2, env.observation_size, env.action_size, lr=3e-4, max_grad_norm=0.5
    )

    # Collect a small rollout
    policies = {
        "predator": PolicyConfig(PolicyType.LEARNED, train_state=pred_state),
        "prey": PolicyConfig(PolicyType.LEARNED, train_state=prey_state),
    }
    rollout_config = RolloutConfig(n_steps=32, n_envs=4)
    k1, key = jax.random.split(key)
    env_keys = jax.random.split(k1, 4)
    obs_init, env_state_init = jax.vmap(env.reset)(env_keys)

    k1, key = jax.random.split(key)
    _, (transitions, infos), _, _ = collect_rollouts(
        env, policies, env_config, rollout_config, k1, obs_init, env_state_init
    )

    # Check transition shapes and reward magnitudes
    for side in ["predator", "prey"]:
        t = transitions[side]
        print(f"\n  {side} transitions:")
        print(f"    obs: {t.obs.shape}")
        print(f"    action: {t.action.shape}")
        print(f"    reward: {t.reward.shape} mean={t.reward.mean():.4f} std={t.reward.std():.4f}")
        print(f"    value: {t.value.shape} mean={t.value.mean():.4f} std={t.value.std():.4f}")
        print(f"    log_prob: {t.log_prob.shape} mean={t.log_prob.mean():.4f}")
        print(f"    done: {t.done.shape} any={t.done.any()}")

    # Run PPO update and check params changed
    for side, state in [("predator", pred_state), ("prey", prey_state)]:
        k1, key = jax.random.split(key)
        old_params = jax.tree.map(lambda x: x.copy(), state.params)

        new_state, metrics, adv_mean, adv_std = ppo_update(
            state,
            transitions[side],
            k1,
            gamma=0.99,
            gae_lambda=0.9,
            clip_eps=0.23,
            vf_coef=1.0,
            ent_coef=0.02,
            n_epochs=4,
            n_minibatches=2,
            normalize_returns=True,
        )

        # Check if params changed
        param_diff = jax.tree.map(lambda a, b: jnp.abs(a - b).max(), old_params, new_state.params)
        max_diff = max(jax.tree.leaves(jax.tree.map(lambda x: x.item(), param_diff)))

        print(f"\n  {side} PPO update:")
        print(f"    adv_mean={adv_mean:.4f} adv_std={adv_std:.4f}")
        print(f"    policy_loss={metrics['policy_loss']:.6f}")
        print(f"    value_loss={metrics['value_loss']:.6f}")
        print(f"    entropy={metrics['entropy']:.4f}")
        print(f"    approx_kl={metrics['approx_kl']:.6f}")
        print(f"    max param change={max_diff:.6f}")
        if max_diff < 1e-7:
            print("    !! PARAMS NOT CHANGING — no gradient flow !!")

    # Check network output magnitudes
    print("\n" + "=" * 60)
    print("6. NETWORK OUTPUT CHECK")
    print("=" * 60)
    sample_obs = transitions["predator"].obs[0, 0, 0]  # [obs_size]
    pred_out = pred_state.apply_fn(pred_state.params, sample_obs)
    print(f"  Pred action_mean: {pred_out.action_mean}")
    print(f"  Pred action_logstd: {pred_out.action_logstd}")
    print(f"  Pred value: {pred_out.value}")

    sample_obs = transitions["prey"].obs[0, 0, 0]
    prey_out = prey_state.apply_fn(prey_state.params, sample_obs)
    print(f"  Prey action_mean: {prey_out.action_mean}")
    print(f"  Prey action_logstd: {prey_out.action_logstd}")
    print(f"  Prey value: {prey_out.value}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)


if __name__ == "__main__":
    main()
