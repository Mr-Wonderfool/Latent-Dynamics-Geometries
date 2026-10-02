import numpy as np
import gymnasium as gym


class CleanObservationWrapper(gym.ObservationWrapper):
    """
    Clean raw observations to only include `qpos` and `qvel`. Also exclude
    global x-y coordinates from `qpos`
    """

    def __init__(self, env):
        super().__init__(env)

        new_shape = env.observation_space.shape
        if "Ant" in env.spec.id:
            new_shape = (27,)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=new_shape, dtype=np.float64)

    def observation(self, observation):
        if "Ant" in self.env.spec.id:
            return observation[:27]
        return observation
