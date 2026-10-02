import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation as R


class BodyFrameWrapper(gym.ObservationWrapper):
    """
    Converts the Ant's Root Linear Velocity from World Frame to Body Frame, preventing split cluster issue
    in latent space caused by directional symmetry.
    """

    def __init__(self, env):
        super().__init__(env)
        # Indices in standard Ant-v2/v3/v4/v5 (excluding x,y):
        # 0: z-pos
        # 1-4: quaternion (w, x, y, z) -> Gym standard usually [w, x, y, z]
        # 5-12: joint angles
        # 13-15: root linear vel (vx, vy, vz) in WORLD frame <-- TARGET
        # 16-18: root angular vel
        # 19-26: joint vels

    def observation(self, obs):
        quat_mujoco = obs[1:5]  # [w, x, y, z]
        # Convert to Scipy format [x, y, z, w]
        quat_scipy = np.array([quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0]])

        # extract world velocity
        vel_world = obs[13:16]

        # create rotation from body to world
        r = R.from_quat(quat_scipy)

        # rotate world velocity into body frame
        vel_body = r.inv().apply(vel_world)

        new_obs = obs.copy()
        new_obs[13:16] = vel_body

        return new_obs
