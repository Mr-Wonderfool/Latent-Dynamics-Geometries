import gymnasium as gym
import mujoco


class LimitAngleTorqueWrapper(gym.Wrapper):
    """
    Limits control range (torque) of the ankle joints to prevent hopping (for environments like Walker2d)
    """

    def __init__(self, env):
        super().__init__(env)
        model = self.unwrapped.model
        ankle_indices = []
        for i in range(model.nu):
            joint_id = model.actuator_trnid[i, 0]
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
            if "foot" in joint_name:
                ankle_indices.append(i)

        for idx in ankle_indices:
            # scale ctrlrange to 20% of the original upper bound
            upper = model.actuator_ctrlrange[idx, 1]
            new_limit = 0.2 * upper
            model.actuator_ctrlrange[idx] = [-new_limit, new_limit]
