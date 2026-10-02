import torch
from torch import nn
from typing import List
import torch.nn.utils.parametrizations as param

from ..models.utils import buildMLP
from ..models.modules.tcn import TCN
from ..models.feature_extractor import feature_extractor_factory


# hard constraint: lipschitz normalization
def applyLipschitzRegularization(module: nn.Module, reg_mode: str):
    """
    Recursively applies Spectral or Orthogonal parametrization to Linear and Conv1d layers.
    """
    if reg_mode not in ["spectral", "orthogonal", "none"]:
        raise ValueError(f"Invalid reg_mode '{reg_mode}'. Use 'spectral', 'orthogonal', or 'none'.")

    if "none" == reg_mode:
        return module

    for child in module.children():
        # If it's a target layer, apply the parametrization
        if isinstance(child, (nn.Linear, nn.Conv1d)):
            if reg_mode == "spectral":
                # Applies Power Iteration approximation
                param.spectral_norm(child, name="weight")
            elif reg_mode == "orthogonal":
                # Applies strict orthogonal constraints (Cayley/Exp)
                param.orthogonal(child, name="weight")
        else:
            # Recursively search through sub-modules (like Sequential)
            applyLipschitzRegularization(child, reg_mode)

    return module


class Encoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        history_len: int,
        embed_dim: int,
        kernel_size_list: List[int],
        stride_list: List[int],
        latent_log_std_min: float = -20.0,
        latent_log_std_max: float = 2.0,
        reg_mode: str = "none",
    ):
        super().__init__()
        input_dim = obs_dim + action_dim
        # embed state action pair
        embed_mlp_dims = [input_dim, embed_dim, embed_dim]
        self.embedding = buildMLP(layer_shape=embed_mlp_dims, activation=nn.ReLU, output_activation=nn.Identity)

        self.tcn = TCN(embed_dim=embed_dim, kernel_size_list=kernel_size_list, stride_list=stride_list)

        # calculate output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, history_len, input_dim)
            # (batch, seq, feature) -> (batch, feature, seq)
            embedded = self.embedding(dummy_input).permute(0, 2, 1)
            tcn_output = self.tcn(embedded)
            flattend_size = tcn_output.reshape(1, -1).size(1)

        self.latent_mean = nn.Linear(flattend_size, latent_dim)
        self.latent_log_std = nn.Linear(flattend_size, latent_dim)

        self.latent_log_std_min = latent_log_std_min
        self.latent_log_std_max = latent_log_std_max

        # apply regularization
        self.embedding = applyLipschitzRegularization(self.embedding, reg_mode)
        self.tcn = applyLipschitzRegularization(self.tcn, reg_mode)
        self.latent_mean = applyLipschitzRegularization(self.latent_mean, reg_mode)
        self.latent_log_std = applyLipschitzRegularization(self.latent_log_std, reg_mode)

    def forward(self, states_with_actions: torch.Tensor) -> torch.distributions.Normal:
        # flip input sequence from [new, ..., old] to [old, ..., new]
        states_with_actions_chron = torch.flip(states_with_actions, dims=[1])
        # (batch, seq, embed_dim) -> (batch, embed_dim, seq)
        embedded = self.embedding(states_with_actions_chron).permute(0, 2, 1)
        tcn_out = self.tcn(embedded)
        flattened = tcn_out.reshape(tcn_out.size(0), -1)
        latent_mean = self.latent_mean(flattened)
        latent_log_std = self.latent_log_std(flattened)
        latent_log_std = torch.clamp(latent_log_std, self.latent_log_std_min, self.latent_log_std_max)
        return torch.distributions.Normal(latent_mean, torch.exp(latent_log_std))


class RunningMeanStd(nn.Module):
    def __init__(self, shape, epsilon: float = 1e-5, clip_limit: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(1e-4))
        self.epsilon = epsilon
        self.clip_limit = clip_limit
        self.shape = shape

    def update(self, x):
        """
        Update inner mean and std using current batch data
        """
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean[:] = new_mean
        self.var[:] = new_var
        self.count.copy_(tot_count)

    def normalize(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + self.epsilon), -self.clip_limit, self.clip_limit)

    def denormalize(self, x):
        return (x * torch.sqrt(self.var + self.epsilon)) + self.mean


class Decoder(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, log_std_min: float = -20.0, log_std_max: float = 2.0, **decoder_kwargs
    ):
        super().__init__()

        self.backbone = feature_extractor_factory[decoder_kwargs["name"]](
            obs_dim=obs_dim + action_dim, **decoder_kwargs["kwargs"]
        )

        # predict the normalized delta state
        feature_dim = self.backbone.feature_dim
        self.prediction_mean = nn.Linear(feature_dim, obs_dim)
        self.prediction_log_std = nn.Linear(feature_dim, obs_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.target_scaler = RunningMeanStd(shape=(obs_dim,))

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor, latent_vecs: torch.Tensor
    ) -> torch.distributions.Normal:
        """
        Predicts the distribution of the normalized change in state.
        """
        combined_inputs = torch.cat([actions, observations, latent_vecs], dim=-1)

        features = self.backbone(combined_inputs)

        state_mean = self.prediction_mean(features)
        state_log_std = self.prediction_log_std(features)
        state_log_std = torch.clamp(state_log_std, self.log_std_min, self.log_std_max)

        return torch.distributions.Normal(state_mean, torch.exp(state_log_std))

    def logProb(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        actions: torch.Tensor,
        latent_vecs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes log prob of the actual transition, handling target normalization internally.
        """
        delta_state = next_observations - observations

        # update scaler statistics (training only)
        if self.training:
            self.target_scaler.update(delta_state.detach())

        normalized_target = self.target_scaler.normalize(delta_state)

        normalized_dist = self(observations=observations, actions=actions, latent_vecs=latent_vecs)

        return normalized_dist.log_prob(normalized_target)

    @torch.no_grad()
    def predictNextObs(self, observations, actions, latent_vecs):
        """
        Return the denormalized next state prediction, along with std
        """
        dist = self(observations, actions, latent_vecs)
        normalized_delta = dist.mean
        normalized_scale = dist.scale

        real_delta = self.target_scaler.denormalize(normalized_delta)
        # denormalize scale
        real_scale = normalized_scale * (self.target_scaler.var + self.target_scaler.epsilon) ** 2

        return observations + real_delta, real_scale
