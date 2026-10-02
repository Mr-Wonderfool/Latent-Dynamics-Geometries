import torch
from torch import nn
from typing import List
from copy import deepcopy

from ldr.common.base_net import BaseFeatureExtractor, BasePolicy


class ContinuousCritic(BasePolicy):
    """
    Critic network predicting Q(s, a) (Q-function), takes as input the
    continuous action and observation, and output two Q-value estimates
    (double Q net in TD3 and SAC)
    """

    def __init__(
        self,
        obs_dim: int,
        feature_extractor: BaseFeatureExtractor,
        latent_vf_extractor: BaseFeatureExtractor,
        n_critics: int = 2,
        share_feature_extractor: bool = False,
    ):
        super().__init__(obs_dim=obs_dim, feature_extractor=feature_extractor)
        self.feature_extractor = feature_extractor
        self.share_feature_extractor = share_feature_extractor
        self.n_critics = n_critics

        self.q_nets: List[nn.Module] = []
        for idx in range(n_critics):
            current_latent_vf_extractor = deepcopy(latent_vf_extractor)
            map_to_q = nn.Linear(latent_vf_extractor.feature_dim, 1)
            q_net = nn.Sequential(current_latent_vf_extractor, map_to_q)
            self.add_module(f"qf{idx}", q_net)
            self.q_nets.append(q_net)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        # ! learn the feature extractor with policy loss only
        # ! when sharing extractor between actor and critic
        with torch.set_grad_enabled(not self.share_feature_extractor):
            features = self.feature_extractor(obs)
        # ! since observation might contain additional information that needs to be
        # ! sliced, we put action in the front and obs in the back
        qvalue_input = torch.cat([action, features], dim=-1)
        return tuple(q_net(qvalue_input) for q_net in self.q_nets)

    def forwardQ1(self, obs: torch.Tensor, action: torch.Tensor):
        """
        Only forward the first critic, used for updating actor network.
        Notice action is from actor which has gradient tracking,
        but feature extractor from critic should not receive gradient
        """
        with torch.no_grad():
            features = self.feature_extractor(obs)
        # ! enable gradient tracking
        return self.q_nets[0](torch.cat([features, action], dim=1))
