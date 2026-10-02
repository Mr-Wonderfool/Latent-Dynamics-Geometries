import torch
from typing import Type

from ..sac_agent.policies import SACPolicy
from ldr.common.schedule import BaseSchedule
from ldr.common.base_net import BaseFeatureExtractor
from ..models.feature_extractor import EnvironmentFactorEncoder


class RMASACPolicy(SACPolicy):
    """
    RMA-adapted SAC Policy.
    Automatically configures the `EnvironmentFactorEncoder` as the feature extractor.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr_schedule: BaseSchedule,
        feature_extractor_class: Type[BaseFeatureExtractor],
        feature_extractor_kwargs: dict,
        optimizer_class: Type[torch.optim.Optimizer],
        optimizer_kwargs: dict,
        device: torch.device,
        **policy_kwargs,
    ):
        assert (
            feature_extractor_class == EnvironmentFactorEncoder
        ), f"RMAAgent should use `EnvironmentFactorEncoder` as feature extractor!"

        factor_dim = policy_kwargs["feature_extractor"]["kwargs"]["factor_dim"]

        super().__init__(
            obs_dim=obs_dim - factor_dim,
            action_dim=action_dim,
            lr_schedule=lr_schedule,
            feature_extractor_class=feature_extractor_class,
            feature_extractor_kwargs=feature_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            device=device,
            **policy_kwargs,
        )
