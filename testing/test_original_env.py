"""Test trial_099 config on 5v5 environment."""

import json
from pathlib import Path
from jax_boids.envs.types import EnvConfig
from jax_boids.train_single import TrainConfig, train

# trial_099 from expanded random sweep (stable, prey=1.415 on simple env)
TRIAL_099_CONFIG = {
    "lr": 0.0013241303945852139,
    "clip": 0.21928748256656988,
    "ent_coef": 0.09072915813460347,
    "vf_coef": 0.3680569568838165,
    "gae_lambda": 0.8951874186071114,
    "max_grad_norm": 0.39194460287058386,
    "n_steps": 64,
    "n_epochs": 2,
    "min_lr": 0.00039723911837556415,
}

# 5v5 environment settings
ENV_5V5 = EnvConfig(
    n_predators=5,
    n_prey=5,
    world_size=20.0,
    max_steps=200,
    prey_learn=False,
    distance_reward=False,
)

OUTPUT_DIR = Path("runs/validate_trial_099_5v5")


def run_config() -> dict:
    """Run trial_099 on 5v5 environment."""
    run_dir = OUTPUT_DIR / "trial_099"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        n_steps=TRIAL_099_CONFIG["n_steps"],
        n_epochs=TRIAL_099_CONFIG["n_epochs"],
        n_envs=16,
        lr=TRIAL_099_CONFIG["lr"],
        clip_eps=TRIAL_099_CONFIG["clip"],
        ent_coef=TRIAL_099_CONFIG["ent_coef"],
        vf_coef=TRIAL_099_CONFIG["vf_coef"],
        gae_lambda=TRIAL_099_CONFIG["gae_lambda"],
        max_grad_norm=TRIAL_099_CONFIG["max_grad_norm"],
        min_lr=TRIAL_099_CONFIG["min_lr"],
        total_timesteps=int(1e6),
        orthogonal_init=True,
        lr_anneal=True,
        normalize_returns=True,
    )

    _, metrics = train(config, ENV_5V5, seed=42, verbose=True, log_dir=str(run_dir))

    # Extract final metrics
    result = {
        "prey_alive": float(metrics["prey_alive"][-1]),
        "kl": float(metrics["approx_kl"][-1]),
        "reward": float(metrics["reward"][-1]),
    }

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(
            {
                "name": "trial_099",
                "lr": TRIAL_099_CONFIG["lr"],
                "clip": TRIAL_099_CONFIG["clip"],
                "ent_coef": TRIAL_099_CONFIG["ent_coef"],
                "vf_coef": TRIAL_099_CONFIG["vf_coef"],
                "gae_lambda": TRIAL_099_CONFIG["gae_lambda"],
                "max_grad_norm": TRIAL_099_CONFIG["max_grad_norm"],
                "n_steps": TRIAL_099_CONFIG["n_steps"],
                "n_epochs": TRIAL_099_CONFIG["n_epochs"],
                "min_lr": TRIAL_099_CONFIG["min_lr"],
                "n_predators": ENV_5V5.n_predators,
                "n_prey": ENV_5V5.n_prey,
                "world_size": ENV_5V5.world_size,
                "total_timesteps": config.total_timesteps,
            },
            f,
            indent=2,
        )

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Validating trial_099 on 5v5 environment")
    print(
        f"LR={TRIAL_099_CONFIG['lr']:.2e}, Clip={TRIAL_099_CONFIG['clip']:.3f}, Ent={TRIAL_099_CONFIG['ent_coef']:.3f}"
    )
    print(f"n_steps={TRIAL_099_CONFIG['n_steps']}, n_epochs={TRIAL_099_CONFIG['n_epochs']}")
    print(f"{'=' * 60}")

    result = run_config()

    # Save result
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump({"trial_099": result}, f, indent=2)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"trial_099       Prey: {result['prey_alive']:.2f}  KL: {result['kl']:8.4f}")

    print(f"\nResults saved to {OUTPUT_DIR}/results.json")


if __name__ == "__main__":
    main()
