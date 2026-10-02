import torch
from pathlib import Path

from ldr.agents import makeAgent
from ldr.agents.rma_agent import RMAAdaptModuleConfig
from ldr.common.utils import Logger, Recorder, ParamsManager
from ldr.environments import GymEnvConfig, makeVectorizedEnv


class RMAAdapter:
    """
    Train the adaptation network for RMA.
    """

    def __init__(self, config_file: dict, pretrained_weight_path: str):
        agent_yaml_path = config_file["agent"]
        env_yaml_path = config_file["environment"]
        agent_adapt_yaml_path = config_file["adapt_agent"]
        agent_params = ParamsManager.parse(agent_yaml_path)
        env_params = ParamsManager.parse(env_yaml_path)
        agent_adapt_params = ParamsManager.parse(agent_adapt_yaml_path)

        agent_name = agent_params["agent_name"]
        assert "RMAAgent" == agent_name
        self.seed = agent_params["pipeline"]["train"]["seed"]

        """ env initialization """
        # make sure enough expert z for adaptation network
        has_factor_wrapper = False
        wrapper_args = env_params["wrappers"]
        local_env_wrapper = wrapper_args["local_env"]
        for wrapper_name in list(local_env_wrapper.keys()):
            if "DynamicsWrapper" in wrapper_name:
                local_env_wrapper[wrapper_name]["required_experience_length"] = 1
            elif "ExtractEnvFactorWrapper" in wrapper_name:
                has_factor_wrapper = True
        assert has_factor_wrapper, "RMA needs `ExtractEnvFactorWrapper` but is not provided!"
        env_config: GymEnvConfig = GymEnvConfig.fromDict(env_params)
        self.venv = makeVectorizedEnv(env_config=env_config)

        """ Logger and recorder initialization """
        weight_path_cls = Path(pretrained_weight_path)
        org_dir = weight_path_cls.parent.parent
        self.logger = Logger(log_level="INFO", resume=True, log_dir=org_dir, backup_list=[agent_adapt_yaml_path])

        self.recorder = Recorder(self.logger.tb_dir)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.agent = makeAgent(
            agent_params=agent_params, env=self.venv, logger=self.logger, recorder=self.recorder, device=self.device
        )
        self.agent.load(pretrained_weight_path)

        """ Adaptation config """
        self.adapt_config = RMAAdaptModuleConfig.fromDict(agent_adapt_params)

        # Seed Everything
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def adapt(self):
        adapt_weight_path = self.agent.adaptWithModule(
            config=self.adapt_config, save_tag="module_adapt", seed=self.seed
        )
        return adapt_weight_path
