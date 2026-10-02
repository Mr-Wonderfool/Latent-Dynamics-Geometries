import numpy as np
import gymnasium as gym

from ..base_wrappers.base_physics_wrapper import BasePhysicsWrapper


class ExtractEnvFactorWrapper(gym.ObservationWrapper):
    """
    Augments the environment observation with explicit dynamics parameters
    captured from the underlying PhysicsWrapper.
    New Obs = [Original Obs, Dynamics Parameters]
    """

    def __init__(self, env):
        super().__init__(env)

        # Locate the physics wrapper in the environment stack
        self.physics_wrapper = None
        current_env = env
        while hasattr(current_env, "env"):
            if isinstance(current_env, BasePhysicsWrapper):
                self.physics_wrapper = current_env
                break
            current_env = current_env.env

        if self.physics_wrapper is None:
            raise ValueError("`ExtractEnvFactorWrapper` expects a `BasePhysicsWrapper` in the environment stack.")

        dummy_params = self.physics_wrapper.getRandomizedParameters()
        self.factor_dim = dummy_params.shape[0]

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(env.observation_space.shape[0] + self.factor_dim,), dtype=np.float64
        )

    def observation(self, observation):
        factors = self.physics_wrapper.getRandomizedParameters()
        return np.concatenate([observation, factors])
