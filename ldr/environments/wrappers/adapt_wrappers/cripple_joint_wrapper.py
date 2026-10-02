import numpy as np
import gymnasium as gym


class CrippleJointWrapper(gym.Wrapper):
    """
    Simulate broken motor by setting specified action to 0 after `lag` timesteps.
    """

    def __init__(self, env, indices, lag: int = 0):
        super().__init__(env)
        self.crippled_indices = indices
        self.lag = lag
        self.step_counter = 0

    def reset(self, **kwargs):
        self.step_counter = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.step_counter += 1
        if self.step_counter > self.lag:
            action = np.copy(action)
            action[self.crippled_indices] = 0.0

        return self.env.step(action)
