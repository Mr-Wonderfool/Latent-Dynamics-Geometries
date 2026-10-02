from dataclasses import dataclass

from ..agent_config import AgentConfig


@dataclass
class SACConfig(AgentConfig):
    reward_scale: float = 1.0
    capacity: int = int(1.0e+6)
    tau: float = 5.0e-3
    target_update_interval: int = 1
    train_start_iterations: int = int(1.0e+4)
