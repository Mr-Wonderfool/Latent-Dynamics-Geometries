import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import Optional
from functools import partial

from ldr.agents import makeAgent
from ldr.common.utils import ParamsManager
from ldr.adaptation.adapt_config import DomainAdapationConfig
from ldr.environments import GymEnvConfig, DummyVecEnv, local_wrapper_factory, global_wrapper_factory, makeEnv
from ldr.environments.wrappers.adapt_wrappers import adapt_wrapper_factory


class BaseEvaluator:
    def __init__(
        self,
        config_file: dict,
        pretrained_weight_path: str,
        required_experience_length: int = int(1e9),
        time_varying: bool = False,
    ):
        """
        Parameters:
            required_experience_length: randomization frequency for test environment, if set to 1, then randomization happens every episode,
                if set to int(1e9), then no randomization at all
            time_varying: if True, enable mid-episode dynamics shifts for time-varying evaluation
        """
        agent_yaml_path = config_file["agent"]
        env_yaml_path = config_file["environment"]
        self.agent_adapt_yaml_path = config_file["adapt_agent"]
        env_adapt_yaml_path = config_file.get("adapt_environment")
        agent_params = ParamsManager.parse(agent_yaml_path)
        env_params = ParamsManager.parse(env_yaml_path)
        test_env_params = deepcopy(env_params)

        self.agent_adapt_params = ParamsManager.parse(self.agent_adapt_yaml_path)
        env_adapt_params = None
        if env_adapt_yaml_path:
            env_adapt_params = ParamsManager.parse(env_adapt_yaml_path)

        # modify wrappers: no randomization during evaluation (static train env for agent)
        wrapper_args = env_params["wrappers"]
        local_env_wrapper = wrapper_args["local_env"]
        for wrapper_name in list(local_env_wrapper.keys()):
            if "DynamicsWrapper" in wrapper_name:
                local_env_wrapper[wrapper_name]["required_experience_length"] = int(1e9)
        # prepare test env
        wrapper_args = test_env_params["wrappers"]
        local_env_wrapper = wrapper_args["local_env"]
        for wrapper_name in list(local_env_wrapper.keys()):
            if "DynamicsWrapper" in wrapper_name:
                local_env_wrapper[wrapper_name]["required_experience_length"] = required_experience_length
                if time_varying:
                    local_env_wrapper[wrapper_name]["time_varying"] = True

        env_config: GymEnvConfig = GymEnvConfig.fromDict(env_params)
        env_config.num_envs = 1
        env_config.render = False

        test_env_config: GymEnvConfig = GymEnvConfig.fromDict(test_env_params)
        test_env_config.num_envs = 1
        test_env_config.render = False

        adapt_config = None
        if env_adapt_params:
            adapt_config = DomainAdapationConfig.fromDict(env_adapt_params)

        self.venv = self.makeVectorizedEnv(env_config=env_config)
        self.test_venv = self.makeVectorizedEnv(env_config=test_env_config, adapt_config=adapt_config)

        self.agent = makeAgent(
            agent_params=agent_params,
            env=self.venv,
            logger=None,
            recorder=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
        self.agent.load(pretrained_weight_path)

        # record model and default parameters
        self.model = self.test_venv.envs[0].unwrapped.model
        self.default_mass = self.model.body_mass.copy()
        self.default_friction = self.model.geom_friction.copy()
        self.default_damping = self.model.dof_damping.copy()
        self.torque_scale = 1.0

        # helper attributes
        self.rand_fn = {
            "mass": self._assignMass,
            "damping": self._assignDamping,
            "friction": self._assignSlideFriction,
            "torque_scale": self._assignTorqueScale,
        }

    @classmethod
    def makeVectorizedEnv(
        cls, env_config: GymEnvConfig, adapt_config: Optional[DomainAdapationConfig] = None
    ) -> DummyVecEnv:
        # apply local wrappers for individual environment
        local_wrappers = []
        for local_wrapper_name, local_wrapper_kwargs in env_config.wrappers.local_env.items():
            wrapper_fn = local_wrapper_factory[local_wrapper_name]
            local_wrappers.append(partial(wrapper_fn, **local_wrapper_kwargs))

        # only render the first environment
        first_env_render_mode = None
        if env_config.render:
            first_env_render_mode = "human"

        # additionally add adapt wrappers
        if adapt_config:
            for adapt_wrapper_name, adapt_wrapper_kwargs in adapt_config.adapt_wrappers.items():
                wrapper_fn = adapt_wrapper_factory[adapt_wrapper_name]
                local_wrappers.append(partial(wrapper_fn, **adapt_wrapper_kwargs))

        venv = DummyVecEnv(
            [makeEnv(env_id=env_config.env_id, render_mode=first_env_render_mode, wrappers=local_wrappers)]
            + [makeEnv(env_id=env_config.env_id, wrappers=local_wrappers) for _ in range(env_config.num_envs - 1)]
        )

        # apply global wrapper for vectorized environment
        for global_wrapper_name, global_wrapper_kwargs in env_config.wrappers.vec_env.items():
            venv = global_wrapper_factory[global_wrapper_name](venv=venv, **global_wrapper_kwargs)

        return venv

    def evalOverParameter(self, param_name: str, scan_values: np.ndarray, episode_each: int = 1):
        raise NotImplementedError()

    def evalGivenScenario(self, eval_episodes: int):
        raise NotImplementedError()

    def _evalOverParameter(self, eval_func, param_name: str, scan_values: np.ndarray, episode_each: int = 1):
        """
        Modify given `param_name` over the specified `scan_values`
        Returns:
            Mean reward for each scan value
        """
        rewards = []
        for val in tqdm(scan_values):
            self.rand_fn[param_name](val)
            reward = eval_func(env=self.test_venv, eval_num=episode_each, action_scale=self.torque_scale)
            rewards.append(reward)
            # change parameter back to default value in case we eval over different parameters
            self.rand_fn[param_name](1.0)

        self.venv.close()
        self.test_venv.close()

        return rewards

    def _evalGivenScenario(self, eval_func, eval_episodes: int):
        """
        Evaluate the agent performance in given scenario for certain episodes
        """
        rewards = []

        for i in tqdm(range(eval_episodes)):
            reward = eval_func(env=self.test_venv, eval_num=1, seed=i, action_scale=self.torque_scale)
            tqdm.write(f"reward {i+1}: {reward}")
            rewards.append(reward)

        mean = np.mean(rewards)
        std = np.std(rewards)

        self.venv.close()
        self.test_venv.close()

        return mean, std

    def _assignSlideFriction(self, val: float):
        new_fric = self.default_friction.copy()
        new_fric[:, 0] *= val
        self.model.geom_friction[:] = new_fric

    def _assignMass(self, val: float):
        self.model.body_mass[:] = self.default_mass * val

    def _assignDamping(self, val: float):
        self.model.dof_damping[:] = self.default_damping * val

    def _assignTorqueScale(self, val: float):
        self.torque_scale = val
