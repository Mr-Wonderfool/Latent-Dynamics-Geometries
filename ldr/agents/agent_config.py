from typing import Dict, Any
from dataclasses import dataclass, field

from ldr.common.base_config import BaseConfig, BaseOptimizerConfig


@dataclass
class OptimizerConfig(BaseOptimizerConfig):
    name: str = "Adam"
    scheduler: Dict[str, Any] = field(default_factory=dict)
    kwargs: dict = field(default_factory=dict)


@dataclass
class DebugConfig(BaseConfig):
    log_dir: str = "../../../../logs"


@dataclass
class TrainConfig(BaseConfig):
    seed: int = 1012
    use_gpu: bool = True
    render_mode: str = "human"
    total_iterations: int = int(1.0e6)


@dataclass
class ValidationConfig(BaseConfig):
    evaluation_interval: int = int(1.0e5)


@dataclass
class PipelineConfig(BaseConfig):
    debug: DebugConfig = field(default_factory=DebugConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    validate: ValidationConfig = field(default_factory=ValidationConfig)


@dataclass
class AgentConfig(BaseConfig):
    gradient_steps: int = 1
    batch_size: int = 128
    train_freq: int = 5

    policy: Dict[str, Any] = field(default_factory=dict)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    gamma: float = 0.99
