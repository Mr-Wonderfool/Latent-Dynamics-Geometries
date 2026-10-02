import numpy as np

from ldr.environments.vec_env.base_vec_env import VecEnvWrapper
from ..base_wrappers.base_physics_wrapper import BasePhysicsWrapper


class GlobalDynamicsIDWrapper(VecEnvWrapper):
    MANAGER_DYNAMICS_ID_STR = "dynamics_ids"

    def __init__(self, venv):
        super().__init__(venv)
        self.curr_dynamics_id = 0

        self.env_dynamics_ids = np.full((self.num_envs,), self.curr_dynamics_id, dtype=np.int64)

    def _writeDynamicsIdsIntoInfos(self, curr_infos):
        # use info from individual worker to catch signal for dynamics changes
        for i, info in enumerate(self.venv.reset_infos):
            if info.get(BasePhysicsWrapper.WORKER_PHYSICS_CHANGE_SIGNAL_STR, False):
                self.curr_dynamics_id += 1
                self.env_dynamics_ids[i] = self.curr_dynamics_id

                del info[BasePhysicsWrapper.WORKER_PHYSICS_CHANGE_SIGNAL_STR]

            curr_infos[i][self.MANAGER_DYNAMICS_ID_STR] = self.env_dynamics_ids[i]

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        # dynamics change signal is only set during reset, so we listen to `reset_infos`
        obs, rewards, dones, infos = self.venv.step_wait()
        self._writeDynamicsIdsIntoInfos(infos)
        return obs, rewards, dones, infos
