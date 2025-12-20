import gymnasium as gym
import numpy as np


class StandingRewardWrapper(gym.Wrapper):
    """Reward staying upright and penalize movement."""

    def __init__(self, env, velocity_penalty: float = 1.0):
        super().__init__(env)
        self.velocity_penalty = velocity_penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Penalize x/y velocity to discourage movement
        velocity = self._get_velocity(obs)
        movement_penalty = self.velocity_penalty * np.linalg.norm(velocity[:2])

        # Keep healthy reward, remove forward reward, add movement penalty
        custom_reward = info.get("reward_survive", 0) - movement_penalty

        return obs, custom_reward, terminated, truncated, info

    def _get_velocity(self, obs):
        # Humanoid-v5: indices 22-24 are torso x, y, z velocity
        return obs[22:25]
