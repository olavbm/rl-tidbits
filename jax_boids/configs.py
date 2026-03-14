"""Named hyperparameter configs for validation and iteration."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class Config:
    """PPO training configuration."""

    # Learning
    lr: float
    clip_eps: float
    ent_coef: float
    vf_coef: float
    gae_lambda: float
    max_grad_norm: float

    # Training structure
    n_steps: int
    n_epochs: int

    # Features
    orthogonal_init: bool
    lr_anneal: bool
    min_lr: float
    normalize_returns: bool

    # Metadata
    name: str = ""
    source: str = ""  # e.g., "sweep_trial_005"
    notes: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = self.source


# Validated configs from sweep
CONFIGS: Dict[str, Config] = {
    # Top validated: consistently achieved < 1.5 prey_alive
    "validated_005": Config(
        name="validated_005",
        source="sweep_trial_005",
        lr=0.00011692605258969922,
        clip_eps=0.47149456447062227,
        ent_coef=0.1296070770493187,
        vf_coef=0.7068482542502411,
        gae_lambda=0.8748151349887241,
        max_grad_norm=0.7562141181553143,
        n_steps=128,
        n_epochs=10,
        orthogonal_init=False,
        lr_anneal=False,
        min_lr=0.0,
        normalize_returns=True,
        notes="Consistently validated: 1.500 in both rounds",
    ),
    "validated_043": Config(
        name="validated_043",
        source="sweep_trial_043",
        lr=0.00015647703542619545,
        clip_eps=0.2254939916286735,
        ent_coef=0.1783283083937024,
        vf_coef=0.8767965030988005,
        gae_lambda=0.9280419823034836,
        max_grad_norm=0.761170654680722,
        n_steps=128,
        n_epochs=10,
        orthogonal_init=True,
        lr_anneal=False,
        min_lr=0.0,
        normalize_returns=False,
        notes="Consistently validated: 1.500 in both rounds",
    ),
    # Original best (but didn't validate)
    "original_best": Config(
        name="original_best",
        source="sweep_trial_099",
        lr=0.0013241303945852139,
        clip_eps=0.21928748256656988,
        ent_coef=0.09072915813460347,
        vf_coef=0.3680569568838165,
        gae_lambda=0.8951874186071114,
        max_grad_norm=0.39194460287058386,
        n_steps=64,
        n_epochs=2,
        orthogonal_init=True,
        lr_anneal=True,
        min_lr=0.00039723911837556415,
        normalize_returns=True,
        notes="Best in sweep (1.415) but failed validation (1.500)",
    ),
    # Variance cases (flipped between rounds)
    "variance_007": Config(
        name="variance_007",
        source="sweep_trial_007",
        lr=0.03457264569937168,
        clip_eps=0.3392833415901909,
        ent_coef=0.028574318088412532,
        vf_coef=0.3547227396441297,
        gae_lambda=0.958023402403283,
        max_grad_norm=0.5850795607005614,
        n_steps=512,
        n_epochs=16,
        orthogonal_init=False,
        lr_anneal=False,
        min_lr=0.0,
        normalize_returns=True,
        notes="R1: 1.400, R2: 1.500 - high variance",
    ),
    "variance_032": Config(
        name="variance_032",
        source="sweep_trial_032",
        lr=0.0005473372858361467,
        clip_eps=0.3778452772937169,
        ent_coef=0.06273546283899825,
        vf_coef=0.7046566399796075,
        gae_lambda=0.9241563436520743,
        max_grad_norm=0.44667589001025443,
        n_steps=128,
        n_epochs=8,
        orthogonal_init=True,
        lr_anneal=False,
        min_lr=0.0,
        normalize_returns=True,
        notes="R1: 1.500, R2: 1.400 - high variance",
    ),
    # Big sweep (1M steps, 100 configs) - top performers
    "sweep1m_080": Config(
        name="sweep1m_080",
        source="big_sweep_1m/trial_080",
        lr=0.0027413667918798415,
        clip_eps=0.3288696658418369,
        ent_coef=0.04816595837126261,
        vf_coef=0.9346217620176661,
        gae_lambda=0.8707520694186339,
        max_grad_norm=0.5150349220194979,
        n_steps=256,
        n_epochs=10,
        orthogonal_init=False,
        lr_anneal=False,
        min_lr=0.0,
        normalize_returns=True,
        notes="Best in big_sweep_1m: prey_alive=1.702, KL=0.0034",
    ),
    "sweep1m_037": Config(
        name="sweep1m_037",
        source="big_sweep_1m/trial_037",
        lr=0.001152943538353758,
        clip_eps=0.0920242400340032,
        ent_coef=0.19039993249049506,
        vf_coef=0.426151448895013,
        gae_lambda=0.8950108223227627,
        max_grad_norm=0.8259066051186392,
        n_steps=128,
        n_epochs=2,
        orthogonal_init=True,
        lr_anneal=False,
        min_lr=0.0,
        normalize_returns=True,
        notes="2nd in big_sweep_1m: prey_alive=1.703, KL=0.0005",
    ),
    "sweep1m_059": Config(
        name="sweep1m_059",
        source="big_sweep_1m/trial_059",
        lr=0.0016346917363461492,
        clip_eps=0.4408075387683248,
        ent_coef=0.09817573890475578,
        vf_coef=0.904539801248901,
        gae_lambda=0.9332393118458182,
        max_grad_norm=0.5224572504374592,
        n_steps=512,
        n_epochs=8,
        orthogonal_init=True,
        lr_anneal=True,
        min_lr=0.0004904075209038447,
        normalize_returns=True,
        notes="3rd in big_sweep_1m: prey_alive=1.704, KL=0.0039",
    ),
}


def get_config(name: str) -> Config:
    """Get a config by name."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]


def list_configs() -> None:
    """Print all available configs."""
    print("Available configs:")
    for name, cfg in CONFIGS.items():
        print(f"  {name:20} {cfg.source:25} lr={cfg.lr:.2e} clip={cfg.clip_eps:.2f}")
