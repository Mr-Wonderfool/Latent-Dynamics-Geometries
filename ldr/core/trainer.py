import random
import torch
import numpy as np

from ldr.agents import makeAgent
from ldr.common.utils import Logger, Recorder, ParamsManager
from ldr.environments import GymEnvConfig, makeVectorizedEnv


class Trainer:
    def __init__(self, config_file: dict):
        agent_yaml_path = config_file["agent"]
        env_yaml_path = config_file["environment"]
        agent_params = ParamsManager.parse(agent_yaml_path)
        env_params = ParamsManager.parse(env_yaml_path)

        agent_name = agent_params["agent_name"]
        env_name = env_params["env_id"].split("-")[0]
        seed = agent_params["pipeline"]["train"]["seed"]

        self._seedEverything(seed)

        """ env initialization """
        env_config: GymEnvConfig = GymEnvConfig.fromDict(env_params)
        self.venv = makeVectorizedEnv(env_config=env_config)
        self.venv.seed(seed)
        # create val environments
        env_config.render = False
        # fix dynamics for consistent validation results
        local_wrappers = env_config.wrappers.local_env
        for wrapper_name in local_wrappers.keys():
            if "DynamicsWrapper" in wrapper_name:
                local_wrappers[wrapper_name]["required_experience_length"] = 1
        self.val_env = makeVectorizedEnv(env_config=env_config)
        self.val_env.seed(seed)

        """ Logger and recorder initialization """
        self.logger = Logger(
            log_level="DEBUG",
            resume=False,
            log_dir=agent_params["pipeline"]["debug"]["log_dir"],
            tag=f"{agent_name}_{env_name}",
            backup_list=[agent_yaml_path, env_yaml_path],
        )
        self.recorder = Recorder(self.logger.tb_dir)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.agent = makeAgent(
            agent_params=agent_params, env=self.venv, logger=self.logger, recorder=self.recorder, device=self.device
        )

    @staticmethod
    def _seedEverything(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(self):
        self.agent.setValEnv(self.val_env)
        self.agent.train()
