from typing import Dict, Any
from dataclasses import dataclass, field

from ..agent_config import OptimizerConfig
from ldr.common.base_config import BaseConfig
from ..sac_agent.sac_config import SACConfig


@dataclass
class RMAConfig(SACConfig):
    latent_dim: int = 3


@dataclass
class AdaptNetConfig(BaseConfig):
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class RMAAdaptModuleConfig(BaseConfig):
    buffer_capacity: int = 100_000
    total_timesteps: int = 600_000
    train_freq: int = 16
    gradient_steps: int = 1
    batch_size: int = 256
    train_start_iterations: int = 1000
    adapt_net: AdaptNetConfig = field(default_factory=AdaptNetConfig)


@dataclass
class RMACMAConfig(BaseConfig):
    cma_max_iters: int = 10
    rollout_episodes: int = 2
    init_std: float = 0.5
    # search for parameter scale which is more numerically stable
    lower_bound: float = 0.2
    upper_bound: float = 2.5
