import torch
from torch import nn

from ..utils import buildMLP
from ..modules.film import FiLMLayer
from ldr.common.base_net import BaseFeatureExtractor


class FiLMFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, obs_dim, latent_dim, mlp_dims: list, feature_dim):
        super().__init__(obs_dim, feature_dim)

        mlp_base_layers_shape = [self.obs_dim] + mlp_dims + [feature_dim]
        self.mlp_base = buildMLP(layer_shape=mlp_base_layers_shape, activation=nn.ReLU, output_activation=nn.Identity)
        # film simplified version: modulation at the last hidden layer
        self.norm = nn.LayerNorm(feature_dim)
        self.film_layer = FiLMLayer(latent_dim=latent_dim, feature_dim=feature_dim)
        self.post_film_activation = nn.ReLU()

    def forward(self, obs_with_latent: torch.Tensor) -> torch.Tensor:
        obs, latent_z = obs_with_latent[..., : self.obs_dim], obs_with_latent[..., self.obs_dim :]
        x = self.mlp_base(obs)

        # apply normalization for stability, then modulate with FiLM
        x_norm = self.norm(x)
        x_modulated = self.film_layer(x_norm, latent_z)

        features = self.post_film_activation(x_modulated)
        return features
