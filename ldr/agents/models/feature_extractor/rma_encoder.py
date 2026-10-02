import torch
from torch import nn

from ..utils import buildMLP
from ldr.common.base_net import BaseFeatureExtractor


class EnvironmentFactorEncoder(BaseFeatureExtractor):
    """
    RMA Phase 1 Encoder.
    Takes [State, Dynamics Factors], slices out the factors,
    encodes them into latent z, and returns [State, z].
    """

    def __init__(self, obs_dim: int, factor_dim: int, latent_dim: int, mlp_dims: list):
        super().__init__(obs_dim=obs_dim + factor_dim, feature_dim=obs_dim + latent_dim)

        self.real_obs_dim = obs_dim
        self.factor_dim = factor_dim

        mlp_dims = [factor_dim] + mlp_dims + [latent_dim]
        self.mlp = buildMLP(layer_shape=mlp_dims, activation=nn.ReLU, output_activation=nn.Identity)

    def forward(self, obs: torch.Tensor):
        # obs shape: (Batch, real_obs_dim + factor_dim)
        state = obs[:, : self.real_obs_dim]
        factors = obs[:, self.real_obs_dim :]

        z = self.mlp(factors)

        # Concatenate state + z for downstream Actor/Critic
        return torch.cat([state, z], dim=-1)
