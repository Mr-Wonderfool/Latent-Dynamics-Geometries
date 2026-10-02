import torch
from torch import nn
from typing import Type, Optional

from ldr.common.schedule import BaseSchedule
from ..models.modules.critic import ContinuousCritic
from ..models.feature_extractor import feature_extractor_factory
from ldr.common.base_net import BaseFeatureExtractor, BasePolicy
from ldr.common.distribution import SquashedDiagGaussianDistribution


class Actor(BasePolicy):
    """
    Actor network in SAC, output a distribution over actions
    Parameters:
        log_std_min: lower bound for log std
        log_std_max: upper bound for log std
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        feature_extractor: BaseFeatureExtractor,
        latent_pi_extractor: BaseFeatureExtractor,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__(obs_dim=obs_dim, feature_extractor=feature_extractor)

        self.latent_pi = latent_pi_extractor

        self.action_dist = SquashedDiagGaussianDistribution(action_dim=action_dim)
        self.mu = nn.Linear(self.latent_pi.feature_dim, action_dim)
        self.log_std = nn.Linear(self.latent_pi.feature_dim, action_dim)
        # bounds for log std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def getActionDistParams(self, obs: torch.Tensor):
        features = self.feature_extractor(obs)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)
        log_std = self.log_std(latent_pi)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean_actions, log_std

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        mean_actions, log_std = self.getActionDistParams(obs)
        # construct normal distribution, then squash with tanh
        # notice output actions are within (-1, 1)
        return self.action_dist.actionFromParams(
            mean_actions=mean_actions, log_std=log_std, deterministic=deterministic
        )

    def actionLogProb(self, obs: torch.Tensor):
        mean_actions, log_std = self.getActionDistParams(obs)
        return self.action_dist.logProbFromParams(mean_actions=mean_actions, log_std=log_std)

    def evaluateActions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Evaluate the action chosen (previously) under the current distribution.
        """
        mean_actions, log_std = self.getActionDistParams(obs)
        # construct internal gaussian distribution (actions not squashed yet)
        self.action_dist.probaDistribution(mean_actions=mean_actions, log_std=log_std)
        # query log prob under squashed distribution
        return self.action_dist.logProb(
            actions=actions,
            # explicitly indicate to reverse actions
            gaussian_actions=None,
        )


class SACPolicy(BasePolicy):
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr_schedule: BaseSchedule,
        feature_extractor_class: Type[BaseFeatureExtractor],
        feature_extractor_kwargs: dict,
        optimizer_class: Type[torch.optim.Optimizer],
        optimizer_kwargs: dict,
        device: torch.device,
        **policy_kwargs
    ):
        """
        Parameters:
            n_critics: number of critics, default to 2 to mitigate overestimation
            share_feature_extractor: whether to share feature extractor between
                actor and critic (which saves computation time). Notice when
                sharing extractor, it is only updated by policy loss
        """
        super().__init__(
            obs_dim=obs_dim,
            feature_extractor_class=feature_extractor_class,
            feature_extractor_kwargs=feature_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.device = device
        self.share_feature_extractor = policy_kwargs.get("share_feature_extractor", False)
        # actor and critic configurations
        actor_kwargs = policy_kwargs["actor"]
        critic_kwargs = policy_kwargs["critic"]
        # construct latent_pi and actor
        latent_pi_name = actor_kwargs.pop("name")
        latent_pi_kwargs = actor_kwargs.pop("kwargs")
        actor_kwargs.update(
            {
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "latent_pi_extractor": self._makeLatentExtractor(obs_dim, latent_pi_name, latent_pi_kwargs),
            }
        )
        self.actor_kwargs = actor_kwargs
        # construct latent_vf and critic
        latent_vf_name = critic_kwargs.pop("name")
        latent_vf_kwargs = critic_kwargs.pop("kwargs")
        critic_kwargs.update(
            {
                "obs_dim": obs_dim,
                "n_critics": policy_kwargs.get("n_critics", 2),
                "share_feature_extractor": self.share_feature_extractor,
                "latent_vf_extractor": self._makeLatentExtractor(
                    obs_dim + action_dim, latent_vf_name, latent_vf_kwargs
                ),
            }
        )
        self.critic_kwargs = critic_kwargs

        self._build(lr_schedule)

    def _build(self, lr_schedule: BaseSchedule):
        self.actor = self.makeActor(feature_extractor=None)
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.share_feature_extractor:
            # share feature extractor between actor and critic, but feature extractor
            # should only be optimized by policy loss
            self.critic = self.makeCritic(feature_extractor=self.actor.feature_extractor)
            critic_parameters = [
                param for name, param in self.critic.named_parameters() if "feature_extractor" not in name
            ]
        else:
            # separate feature extractor for critic
            self.critic = self.makeCritic(feature_extractor=None)
            critic_parameters = list(self.critic.parameters())

        self.critic.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)
        self.critic_target = self.makeCritic(feature_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_target.train(False)

    def _makeLatentExtractor(self, obs_dim: int, latent_extractor_name: str, latent_extractor_kwargs: dict):
        latent_extractor_kwargs.update({"obs_dim": obs_dim})
        return feature_extractor_factory[latent_extractor_name](**latent_extractor_kwargs)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        return self.actor(obs, deterministic)

    def makeActor(self, feature_extractor: Optional[BaseFeatureExtractor] = None):
        actor_kwargs = self._updateFeatureExtractor(self.actor_kwargs, feature_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def makeCritic(self, feature_extractor: Optional[BaseFeatureExtractor] = None):
        critic_kwargs = self._updateFeatureExtractor(self.critic_kwargs, feature_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def setTrainingMode(self, train: bool):
        self.actor.train(train)
        self.critic.train(train)
