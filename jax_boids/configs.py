"""Named hyperparameter configs for validation and iteration."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Config:
    """PPO training configuration.

    Prey-specific overrides (prey_*) default to None, meaning use the
    predator value. Only relevant for IPPO training.
    """

    # Learning (predator / default)
    lr: float
    clip_eps: float
    ent_coef: float
    vf_coef: float
    gae_lambda: float
    max_grad_norm: float

    # Training structure
    n_steps: int
    n_epochs: int

    # Scale
    n_envs: int

    # Features
    orthogonal_init: bool
    lr_anneal: bool
    min_lr: float
    normalize_returns: bool

    # Prey-specific overrides (None = use predator value)
    prey_lr: Optional[float] = None
    prey_gamma: Optional[float] = None
    prey_gae_lambda: Optional[float] = None
    prey_clip_eps: Optional[float] = None
    prey_vf_coef: Optional[float] = None
    prey_ent_coef: Optional[float] = None
    prey_max_grad_norm: Optional[float] = None
    prey_orthogonal_init: Optional[bool] = None
    prey_lr_anneal: Optional[bool] = None
    prey_min_lr: Optional[float] = None
    prey_normalize_returns: Optional[bool] = None

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
        n_envs=32,
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
        n_envs=32,
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
        n_envs=32,
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
        n_envs=32,
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
        n_envs=32,
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
        n_envs=64,
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
        n_envs=64,
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
        n_envs=64,
        orthogonal_init=True,
        lr_anneal=True,
        min_lr=0.0004904075209038447,
        normalize_returns=True,
        notes="3rd in big_sweep_1m: prey_alive=1.704, KL=0.0039",
    ),
    # Big sweep r03 (capture_radius=0.3, 1M steps, 100 configs) - top performers
    "sweep_r03_003": Config(
        name="sweep_r03_003",
        source="sweep_r03/trial_003",
        lr=0.0025995856171946224,
        clip_eps=0.25674119585232913,
        ent_coef=0.02496523257064187,
        vf_coef=0.9417215290211199,
        gae_lambda=0.8614260287213764,
        max_grad_norm=0.3638604532663836,
        n_steps=256,
        n_epochs=16,
        n_envs=64,
        orthogonal_init=True,
        lr_anneal=True,
        min_lr=0.0007798756851583867,
        normalize_returns=True,
        notes="Best in sweep_r03 (r=0.3): prey_alive=2.17, KL=0.0038",
    ),
    "sweep_r03_022": Config(
        name="sweep_r03_022",
        source="sweep_r03/trial_022",
        lr=0.004260814802596222,
        clip_eps=0.47061781598139224,
        ent_coef=0.04085183988470729,
        vf_coef=0.7871438505920612,
        gae_lambda=0.8846094631292977,
        max_grad_norm=0.45620726211212903,
        n_steps=256,
        n_epochs=10,
        n_envs=64,
        orthogonal_init=False,
        lr_anneal=False,
        min_lr=0.0,
        normalize_returns=True,
        notes="2nd in sweep_r03 (r=0.3): prey_alive=2.18, KL=0.0077",
    ),
    "sweep_r03_080": Config(
        name="sweep_r03_080",
        source="sweep_r03/trial_080",
        lr=0.0027413667918798415,
        clip_eps=0.3288696658418369,
        ent_coef=0.04816595837126261,
        vf_coef=0.9346217620176661,
        gae_lambda=0.8707520694186339,
        max_grad_norm=0.5150349220194979,
        n_steps=256,
        n_epochs=10,
        n_envs=64,
        orthogonal_init=False,
        lr_anneal=False,
        min_lr=0.0,
        normalize_returns=True,
        notes="3rd in sweep_r03 (r=0.3): prey_alive=2.20, KL=0.0036",
    ),
    # Fine-tuned configs (capture_radius=0.3, 5M steps)
    "best_r03": Config(
        name="best_r03",
        source="fine_tuning_sweep_r03_003/fine_011",
        lr=0.002652900442367417,
        clip_eps=0.28813170816691563,
        ent_coef=0.024402997757694122,
        vf_coef=1.0674575613742217,
        gae_lambda=0.8373347764625199,
        max_grad_norm=0.2547925045246027,
        n_steps=256,
        n_epochs=16,
        n_envs=64,
        orthogonal_init=False,
        lr_anneal=False,
        min_lr=0.0,
        normalize_returns=True,
        notes="Best at r=0.3: prey_alive=2.08, KL=0.039 (high). Fine-tuned from sweep_r03_003.",
    ),
    "best_r03_stable": Config(
        name="best_r03_stable",
        source="fine_tuning_sweep_r03_003/fine_014",
        lr=0.002757922486960815,
        clip_eps=0.2376695437266057,
        ent_coef=0.020374263488738473,
        vf_coef=0.6858570405874247,
        gae_lambda=0.6109012199674343,
        max_grad_norm=0.35888235068408364,
        n_steps=512,
        n_epochs=10,
        n_envs=64,
        orthogonal_init=False,
        lr_anneal=True,
        min_lr=0.0008273767460882445,
        normalize_returns=True,
        notes="2nd best at r=0.3: prey_alive=2.09, KL=0.003. From sweep_r03_003.",
    ),
    # 2-predator configs (shared weights, 5 prey, non-learning)
    "pred2_base": Config(
        name="pred2_base",
        source="derived from best_r03",
        lr=0.0027,
        clip_eps=0.33,
        ent_coef=0.048,
        vf_coef=0.93,
        gae_lambda=0.87,
        max_grad_norm=0.515,
        n_steps=256,
        n_epochs=10,
        n_envs=64,
        orthogonal_init=False,
        lr_anneal=False,
        min_lr=0.0,
        normalize_returns=True,
        notes="Base config for 2 predators, 5 prey. Derived from best_r03.",
    ),
    "best_pred2": Config(
        name="best_pred2",
        source="fine_pred2_003_v2/fine_000",
        lr=0.0026952036273721618,
        clip_eps=0.22962322154656425,
        ent_coef=0.021864072717850163,
        vf_coef=1.0155437306641852,
        gae_lambda=0.9067109111002777,
        max_grad_norm=0.4179553286441464,
        n_steps=128,
        n_epochs=16,
        n_envs=64,
        orthogonal_init=True,
        lr_anneal=False,
        min_lr=0.0,
        normalize_returns=True,
        notes="Best 2-pred config: prey_alive=2.37 at 5M steps.",
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
