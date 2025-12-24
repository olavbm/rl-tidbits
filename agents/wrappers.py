import gymnasium as gym
import numpy as np


class VelocityRewardWrapper(gym.Wrapper):
    """Reward moving fast."""

    def __init__(self, env, velocity_bonus: float = 1.0):
        super().__init__(env)
        self.velocity_bonus = velocity_bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        velocity = obs[22:25]  # x, y, z velocity
        speed = np.linalg.norm(velocity[:2])  # horizontal speed

        custom_reward = info.get("reward_survive", 0) + self.velocity_bonus * speed

        return obs, custom_reward, terminated, truncated, info


class SprintRewardWrapper(gym.Wrapper):
    """Aggressive speed reward - incentivize running, not walking."""

    def __init__(self, env, speed_weight: float = 5.0, survive_weight: float = 0.5):
        super().__init__(env)
        self.speed_weight = speed_weight
        self.survive_weight = survive_weight

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        velocity = obs[22:25]  # x, y, z velocity
        speed = np.linalg.norm(velocity[:2])  # horizontal speed

        # Heavy speed reward, reduced survival bonus
        custom_reward = (
            self.survive_weight * info.get("reward_survive", 0)
            + self.speed_weight * speed
        )

        info["speed"] = speed
        return obs, custom_reward, terminated, truncated, info
