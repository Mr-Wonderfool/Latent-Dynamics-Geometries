import torch
from torch import nn

from .mlp_extractor import MLPFeatureExtractor


class LatentMLPFeatureExtractor(MLPFeatureExtractor):
    """
    MLP feature extractor for observation containing latent vector
    """

    def __init__(self, obs_dim, latent_dim, feature_dim, mlp_dims, latent_layer_norm: bool = True):
        """
        Parameters:
            latent_layer_norm: whether to use `LayerNorm` for latent vector
        """
        super().__init__(obs_dim=obs_dim + latent_dim, feature_dim=feature_dim, mlp_dims=mlp_dims)

        self.latent_layer_norm = latent_layer_norm
        if self.latent_layer_norm:
            self.norm = nn.LayerNorm(latent_dim)

        self.obs_dim = obs_dim

    def forward(self, obs_with_latent: torch.Tensor):
        obs, latent = obs_with_latent[..., : self.obs_dim], obs_with_latent[..., self.obs_dim :]
        if self.latent_layer_norm:
            latent = self.norm(latent)
        normalized_input = torch.cat([obs, latent], dim=-1)

        return self.mlp(normalized_input)
