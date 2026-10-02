from ..common.utils import Recorder, Logger
from ..environments.vec_env.base_vec_env import VecEnv
from .sac_agent import SACConfig, SACAgent
from .ldr_agent import LDRConfig, LDRAgent
from .rma_agent import RMAConfig, RMAAgent

valid_agents = [
    "SACAgent",
    "LDRAgent",
    "RMAAgent",
]


def makeAgent(agent_params: dict, env: VecEnv, logger: Logger, recorder: Recorder, device):
    agent_name = agent_params["agent_name"]
    if "SACAgent" == agent_name:
        return SACAgent(
            env=env, config=SACConfig.fromDict(agent_params), logger=logger, recorder=recorder, device=device
        )
    elif "LDRAgent" == agent_name:
        return LDRAgent(
            env=env, config=LDRConfig.fromDict(agent_params), logger=logger, recorder=recorder, device=device
        )
    elif "RMAAgent" == agent_name:
        from ldr.environments.wrappers.local_wrappers import ExtractEnvFactorWrapper

        # extract factor dim from env
        factor_dim = None
        sample_env = env.envs[0]
        while hasattr(sample_env, "env"):
            if isinstance(sample_env, ExtractEnvFactorWrapper):
                factor_dim = sample_env.factor_dim
                break
            sample_env = sample_env.env

        if factor_dim is None:
            raise ValueError("`RMAAgent` requires the environment to be wrapped with `ExtractEnvFactorWrapper`!")

        agent_params["policy"]["feature_extractor"]["kwargs"].update({"factor_dim": factor_dim})
        return RMAAgent(
            env=env, config=RMAConfig.fromDict(agent_params), logger=logger, recorder=recorder, device=device
        )
    else:
        raise NotImplementedError(f"{agent_name} is not supported, available agents: {valid_agents}")


__all__ = ["valid_agents", "makeAgent"]
