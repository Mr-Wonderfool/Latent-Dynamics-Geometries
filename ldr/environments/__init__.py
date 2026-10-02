import gymnasium as gym
from functools import partial
from typing import List, Optional

from .vec_env import DummyVecEnv
from .gym_env_config import GymEnvConfig
from .wrappers import local_wrapper_factory, global_wrapper_factory


def makeEnv(env_id: str, render_mode: Optional[str] = None, wrappers: List[gym.Wrapper] = []):
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        for wrapper in wrappers:
            env = wrapper(env)

        return env

    return _init


def makeVectorizedEnv(env_config: GymEnvConfig) -> DummyVecEnv:
    # apply local wrappers for individual environment
    local_wrappers = []
    for local_wrapper_name, local_wrapper_kwargs in env_config.wrappers.local_env.items():
        wrapper_fn = local_wrapper_factory[local_wrapper_name]
        local_wrappers.append(partial(wrapper_fn, **local_wrapper_kwargs))

    # only render the first environment
    first_env_render_mode = None
    if env_config.render:
        first_env_render_mode = "human"

    venv = DummyVecEnv(
        [makeEnv(env_id=env_config.env_id, render_mode=first_env_render_mode, wrappers=local_wrappers)]
        + [makeEnv(env_id=env_config.env_id, wrappers=local_wrappers) for _ in range(env_config.num_envs - 1)]
    )

    # apply global wrapper for vectorized environment
    for global_wrapper_name, global_wrapper_kwargs in env_config.wrappers.vec_env.items():
        venv = global_wrapper_factory[global_wrapper_name](venv=venv, **global_wrapper_kwargs)

    return venv


__all__ = ["GymEnvConfig", "makeEnv", "makeVectorizedEnv"]
