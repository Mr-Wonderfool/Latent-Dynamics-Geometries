from typing import Dict, Any
from dataclasses import dataclass, field

from ..agent_config import OptimizerConfig
from ..sac_agent.sac_config import SACConfig
from ldr.common.base_config import BaseConfig


@dataclass
class VAEConfig(BaseConfig):
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class LDRConfig(SACConfig):
    # representation learning
    kl_schedule: Dict[str, Any] = field(default_factory=dict)
    n_past: int = 8
    latent_dim: int = 3
    vae_loss_weight: float = 1.0

    # contrastive learning
    infonce_temp: float = 1.0
    contrastive_start_iterations: int = 0
    contrastive_loss_weight: float = 1.0
    min_positives: int = 1

    encoder: VAEConfig = field(default_factory=VAEConfig)
    decoder: VAEConfig = field(default_factory=VAEConfig)

    # latent dropout
    latent_dropout_prob: float = 0.1

    # directly start training since we have larger batch size
    train_start_iterations: int = 0

    # encoder regularization mode
    encoder_reg_mode: str = "none"


@dataclass
class LDRAdaptConfig(BaseConfig):
    buffer_capacity: int = 1000
    total_timesteps: int = 1000
    train_start_iterations: int = 0
    train_freq: int = 16
    gradient_steps: int = 1
    batch_size: int = 256
    beta: float = 1.0e-4
    lr: float = 1.0e-5
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
