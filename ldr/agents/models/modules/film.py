import torch
from torch import nn


class FiLMLayer(nn.Module):
    def __init__(self, latent_dim: int, feature_dim: int):
        super().__init__()
        # add latent norm to prevent magnitude spikes
        self.latent_norm = nn.LayerNorm(latent_dim)
        # map latent code to weight and bias
        self.controller = nn.Linear(latent_dim, 2 * feature_dim)
        self.feature_dim = feature_dim
        # initialize to be an identity mapping
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            self.controller.weight.data.zero_()
            identity_bias = torch.cat([torch.ones(self.feature_dim), torch.zeros(self.feature_dim)])
            self.controller.bias.data.copy_(identity_bias)

    def forward(self, features: torch.Tensor, latent_z: torch.Tensor):
        z_norm = self.latent_norm(latent_z)
        # generate gamma and beta from z
        gamma_beta = self.controller(z_norm)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        return gamma * features + beta
