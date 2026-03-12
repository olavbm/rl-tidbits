"""Curriculum learning for predator-prey environment."""

from jax_boids.envs.types import CurriculumStage, EnvConfig


def get_default_curriculum() -> list[CurriculumStage]:
    """Return the default 4-stage curriculum.

    Each stage trains for TIMESTEPS_PER_STAGE timesteps.
    """
    return [
        CurriculumStage(
            name="stage1_easy",
            n_prey=5,
            world_size=20.0,
            prey_speed_mult=1.2,
            predator_speed_mult=1.5,
            max_steps=100,
        ),
        CurriculumStage(
            name="stage2_medium",
            n_prey=10,
            world_size=30.0,
            prey_speed_mult=1.0,
            max_steps=200,
        ),
        CurriculumStage(
            name="stage3_hard",
            n_prey=20,
            world_size=100.0,
            prey_speed_mult=1.0,
            max_steps=500,
        ),
        CurriculumStage(
            name="stage4_expert",
            n_prey=50,
            world_size=100.0,
            prey_speed_mult=1.2,
            max_steps=800,
        ),
    ]


TIMESTEPS_PER_STAGE = 500_000


def apply_stage(config: EnvConfig, stage: CurriculumStage) -> EnvConfig:
    """Apply a curriculum stage to the environment config.

    Args:
        config: current environment config
        stage: curriculum stage to apply

    Returns:
        updated config with stage parameters
    """
    return EnvConfig(
        **config.__dict__,
        n_prey=stage.n_prey,
        world_size=stage.world_size,
        max_steps=stage.max_steps,
        prey_speed_mult=stage.prey_speed_mult,
        predator_speed_bonus=1.0 + (stage.predator_speed_mult - 1.0),
    )


def advance_curriculum(config: EnvConfig) -> tuple[EnvConfig, bool]:
    """Advance to next curriculum stage if ready.

    Args:
        config: current environment config with curriculum tracking

    Returns:
        tuple of (updated_config, stage_advanced_flag)
    """
    if config.curriculum is None:
        return config, False

    if config.current_stage >= len(config.curriculum) - 1:
        # Already at last stage
        return config, False

    if config.curriculum_timesteps < TIMESTEPS_PER_STAGE:
        return config, False

    # Advance to next stage
    next_stage_idx = config.current_stage + 1
    next_stage = config.curriculum[next_stage_idx]

    new_config = apply_stage(config, next_stage)
    new_config = EnvConfig(
        **new_config.__dict__,
        current_stage=next_stage_idx,
        curriculum_timesteps=0,
    )

    return new_config, True
