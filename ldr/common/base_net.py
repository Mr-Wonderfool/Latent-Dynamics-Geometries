import torch
from torch import nn
from typing import Optional, Type


class BaseFeatureExtractor(nn.Module):
    def __init__(self, obs_dim, feature_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.feature_dim = feature_dim


class BasePolicy(nn.Module):
    """
    Base class for all networks, including actor and critic
    """

    optimizer: torch.optim.Optimizer
    feature_extractor: BaseFeatureExtractor

    def __init__(
        self,
        obs_dim: int,
        feature_extractor: Optional[BaseFeatureExtractor] = None,
        feature_extractor_class: Type[BaseFeatureExtractor] = BaseFeatureExtractor,
        feature_extractor_kwargs: Optional[dict] = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        if feature_extractor_kwargs is None:
            feature_extractor_kwargs = {}
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        # environment specific parameter
        self.obs_dim = obs_dim

        self.feature_extractor = feature_extractor
        self.feature_extractor_class = feature_extractor_class
        self.feature_extractor_kwargs = feature_extractor_kwargs

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

    def _updateFeatureExtractor(self, net_kwargs: dict, feature_extractor: Optional[BaseFeatureExtractor] = None):
        net_kwargs = net_kwargs.copy()
        if feature_extractor is None:
            feature_extractor = self.makeFeatureExtractor()
        net_kwargs.update(dict(feature_extractor=feature_extractor))
        return net_kwargs

    def makeFeatureExtractor(self):
        return self.feature_extractor_class(obs_dim=self.obs_dim, **self.feature_extractor_kwargs)

    def predict(self, obs: torch.Tensor, deterministic: bool = True):
        raise NotImplementedError()
