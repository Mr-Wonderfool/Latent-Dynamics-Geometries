import os

from ldr.common.utils import ParamsManager


def modelBase(erase_dynamics_wrappers: bool = True, model_path: str = ""):
    config_path = os.path.abspath(os.path.join(__file__, "../../configs/user_config.yaml"))
    config_basedir_path = os.path.dirname(config_path)
    # change relative path to absolute
    config_file = ParamsManager.load(config_path)
    config_file["agent"] = os.path.join(config_basedir_path, config_file["agent"])
    config_file["environment"] = os.path.join(config_basedir_path, config_file["environment"])

    agent_params = ParamsManager.parse(config_file["agent"])
    env_params = ParamsManager.parse(config_file["environment"])

    agent_params.setdefault("pipeline", {})
    agent_params["pipeline"]["test"] = {"model": model_path}

    wrapper_args = env_params["wrappers"]
    if erase_dynamics_wrappers:
        # erase dynamics randomization wrapper for eval consistency

        local_env_wrapper = wrapper_args["local_env"]
        for wrapper_name in list(local_env_wrapper.keys()):
            if "DynamicsWrapper" in wrapper_name:
                del local_env_wrapper[wrapper_name]

        global_env_wrapper = wrapper_args["vec_env"]
        for wrapper_name in list(global_env_wrapper.keys()):
            if "GlobalDynamicsIDWrapper" in wrapper_name:
                del global_env_wrapper[wrapper_name]
    else:
        # for dynamics evaluation, we want randomization at every episode
        local_env_wrapper = wrapper_args["local_env"]
        for wrapper_name in list(local_env_wrapper.keys()):
            if "DynamicsWrapper" in wrapper_name:
                local_env_wrapper[wrapper_name]["required_experience_length"] = 1
                break

    return env_params, agent_params
