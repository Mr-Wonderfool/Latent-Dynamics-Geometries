import torch.nn as nn

from ..utils import buildMLP
from ldr.common.base_net import BaseFeatureExtractor


class MLPFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, obs_dim, feature_dim, mlp_dims: list):
        super().__init__(obs_dim=obs_dim, feature_dim=feature_dim)

        mlp_dims = [obs_dim] + mlp_dims + [feature_dim]
        self.mlp = buildMLP(layer_shape=mlp_dims, activation=nn.ReLU, output_activation=nn.ReLU)

    def forward(self, obs):
        # obs with shape (n_envs, obs_dim)
        return self.mlp(obs)
