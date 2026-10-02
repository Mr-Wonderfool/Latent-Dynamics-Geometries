from typing import Dict
from dataclasses import dataclass, field

from ldr.common.base_config import BaseConfig


@dataclass
class WrapperConfig(BaseConfig):
    local_env: Dict[str, Dict] = field(default_factory=dict)
    vec_env: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class GymEnvConfig(BaseConfig):
    env_id: str = "Hopper-v5"
    render: bool = False
    num_envs: int = 1
    wrappers: WrapperConfig = field(default_factory=WrapperConfig)
