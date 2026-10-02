import torch
from torch import nn
from typing import Optional
from torch.distributions import Normal

from .base import Distribution


def sumIndependentActionDims(tensor: torch.Tensor):
    # input shape (batch, ..., n_actions) or (batch, n_actions)
    if len(tensor.shape) > 1:
        return tensor.sum(dim=-1)


class Tanh:
    """
    Provides access to tanh and the inverse of tanh, in a numerically stable way
    """

    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    @staticmethod
    def forward(x: torch.Tensor):
        return torch.tanh(x)

    @staticmethod
    def atanh(x: torch.Tensor):
        """
        Inverse of tanh, 0.5*(log(1 + x) - log(1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(x: torch.Tensor):
        eps = torch.finfo(x.dtype).eps
        # clip to avoid NaN
        return Tanh.atanh(x.clamp(min=-1.0 + eps, max=1.0 - eps))


class DiagGaussianDistribution(Distribution):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim
        # mean actions and std
        self.mean_actions = None
        self.log_std = None

    def probaDistributionNet(self, latent_dim: int, log_std_init: float = 0.0):
        dist_net = nn.Linear(latent_dim, self.action_dim)
        log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init, requires_grad=True)
        return dist_net, log_std

    def probaDistribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor):
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def sample(self):
        """Reparameterization trick to pass gradients"""
        return self.distribution.rsample()

    def logProb(self, actions: torch.Tensor):
        log_prob = self.distribution.log_prob(actions)
        return sumIndependentActionDims(log_prob)

    def mode(self):
        return self.distribution.mean

    def entropy(self):
        return sumIndependentActionDims(self.distribution.entropy())

    def actionFromParams(self, mean_actions: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False):
        self.probaDistribution(mean_actions=mean_actions, log_std=log_std)
        return self.getActions(deterministic)


class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    def __init__(self, action_dim):
        super().__init__(action_dim)
        self.gaussian_actions: torch.Tensor = None
        self.epsilon = 1e-6

    def probaDistribution(self, mean_actions, log_std):
        super().probaDistribution(mean_actions, log_std)
        return self

    def logProb(self, actions: torch.Tensor, gaussian_actions: Optional[torch.Tensor] = None):
        """
        Log probability of the squashed distribution considering transformation between random variables
        Parameters:
            actions: squashed actions in (-1, 1), shape (batch, action_dim)
            gaussian_actions: actions from original gaussian distribution, shape (batch, action_dim)
        """
        if gaussian_actions is None:
            # inverse tanh to produce original gaussian actions
            gaussian_actions = Tanh.inverse(actions)
        log_prob = super().logProb(gaussian_actions)
        # apply transformation between random variables
        log_prob -= torch.sum(torch.log(1 - actions**2 + self.epsilon), dim=-1)
        return log_prob

    def entropy(self):
        """
        No analytical expression for entropy, and has to be estimated via -sum(log_probs).
        """
        return None

    def sample(self):
        self.gaussian_actions = super().sample()
        return torch.tanh(self.gaussian_actions)

    def mode(self):
        self.gaussian_actions = super().mode()
        return torch.tanh(self.gaussian_actions)

    def logProbFromParams(self, mean_actions: torch.Tensor, log_std: torch.Tensor):
        # sample actions using reparameterization trick
        action = self.actionFromParams(mean_actions, log_std)
        log_prob = self.logProb(actions=action, gaussian_actions=self.gaussian_actions)
        return action, log_prob
