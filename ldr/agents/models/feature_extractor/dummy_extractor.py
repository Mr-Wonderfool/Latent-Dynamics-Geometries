import torch
import torch.nn as nn

from ldr.common.base_net import BaseFeatureExtractor


class DummyExtractor(BaseFeatureExtractor):
    def __init__(self, obs_dim):
        super().__init__(obs_dim=obs_dim, feature_dim=obs_dim)
        self.extractor = nn.Identity()

    def forward(self, observation: torch.Tensor):
        return self.extractor(observation)
