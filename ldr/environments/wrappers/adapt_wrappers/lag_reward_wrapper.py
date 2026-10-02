import gymnasium as gym


class LagRewardWrapper(gym.Wrapper):
    def __init__(self, env, lag: int):
        super().__init__(env)
        self.lag = lag
        self.step_counter = 0

    def reset(self, *, seed=None, options=None):
        self.step_counter = 0

        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.step_counter += 1

        # Mask the reward if haven't reached the lag threshold
        if self.step_counter <= self.lag:
            reward = 0.0

        return obs, reward, terminated, truncated, info
