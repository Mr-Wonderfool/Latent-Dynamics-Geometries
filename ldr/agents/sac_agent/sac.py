import torch
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from typing import Type
import torch.nn.functional as F

from ..agent import Agent
from .sac_config import SACConfig
from ldr.common.utils import Logger, Recorder
from ..utils.utils import toTensor, polyakUpdate
from .policies import SACPolicy, Actor, ContinuousCritic
from ldr.environments.vec_env.base_vec_env import VecEnv
from ..models.feature_extractor import feature_extractor_factory
from ldr.common.buffer.replay_buffer import ReplayBuffer, ReplayBufferSamples


class SACAgent(Agent):

    policy: SACPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic
    config: SACConfig

    def __init__(
        self,
        env: VecEnv,
        config: SACConfig,
        logger: Logger,
        recorder: Recorder,
        device: torch.device,
        _policy_cls: Type[SACPolicy] = SACPolicy,
        _init_setup_buffer: bool = True,
    ):
        """
        Note:
            entropy coefficient is chosen automatically, as in https://arxiv.org/pdf/1812.05905
            target entropy is set to the negated dimension of action space by default
        """
        super().__init__(env=env, config=config, seed=config.pipeline.train.seed, logger=logger, recorder=recorder)
        assert isinstance(self.train_env.action_space, gym.spaces.Box), "Using Continuous SAC on discrete action space!"
        # record action space lower and upper bound for action scaling
        self.action_low = self.train_env.action_space.low
        self.action_high = self.train_env.action_space.high
        self.device = device

        # policy configurations
        self.optimizer_class = self.config.optimizer.optimizerFromName(self.config.optimizer.name)
        scheduler_kwargs = self.config.optimizer.scheduler
        self.lr_schedule = self.config.optimizer.schedulerFromName(
            scheduler_name=scheduler_kwargs.pop("name"), scheduler_kwargs=scheduler_kwargs
        )
        feature_extractor_name = self.config.policy["feature_extractor"]["name"]
        feature_extractor_kwargs = self.config.policy["feature_extractor"]["kwargs"]
        self.policy = _policy_cls(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            lr_schedule=self.lr_schedule,
            feature_extractor_class=feature_extractor_factory[feature_extractor_name],
            feature_extractor_kwargs=feature_extractor_kwargs,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.config.optimizer.kwargs,
            device=device,
            **self.config.policy,
        )
        self.policy = self.policy.to(device)
        # create alias for actor and critic
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

        # sac specific temperature parameter (adjusted automatically)
        self.target_entropy = float(-np.prod(self.train_env.action_space.shape).astype(np.float32))
        init_ent_coef = 1.0
        self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * init_ent_coef).requires_grad_(True)
        self.ent_coef_optimizer = self.optimizer_class([self.log_ent_coef], lr=self.lr_schedule(1))
        self.ent_coef = None

        # training configurations
        self.total_iterations = self.config.pipeline.train.total_iterations

        # model saving utils
        self.best_reward = float("-inf")

        # adjust train parameters
        scale_factor = max(self.config.train_freq // self.env_dim, 1)
        evaluation_interval = max(self.config.pipeline.validate.evaluation_interval // self.env_dim, 1)
        self.evaluation_interval = max(evaluation_interval // scale_factor * scale_factor, 1)

        # when stacking observations, buffer should be initialized in child class
        if _init_setup_buffer:
            self.replay_buffer = ReplayBuffer(
                env_dim=self.env_dim, obs_dim=self.obs_dim, action_dim=self.action_dim, capacity=self.config.capacity
            )

    def train(self):
        self._onTrainingStarts()
        with tqdm(
            total=self.total_iterations,
            desc="Training",
            unit="step",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            while self._num_timesteps < self.total_iterations:
                """interaction with environment"""
                num_steps = self.explore()
                pbar.update(num_steps)

                """train the model"""
                if self._num_timesteps >= self.config.train_start_iterations:
                    self.recorder.record(
                        "train/learning rate", self.lr_schedule(self._current_progress_remaining), self._num_timesteps
                    )
                    self.optimize()

                    """evaluate the model"""
                    if self._n_calls % self.evaluation_interval == 0:
                        rewards_val = self.evaluate(self.val_env, 1)

                        """logging"""
                        timesteps_k = round(self._num_timesteps / 1e3, 2)
                        self.recorder.record("test/val reward", rewards_val, self._num_timesteps)
                        msg = f"iteration {timesteps_k}k has reward [Val]: {rewards_val:.2f}"
                        self.logger.info(msg)
                        tqdm.write(msg)
                        # save model
                        if rewards_val > self.best_reward:
                            msg = f"new best reward {rewards_val:.2f} found at {timesteps_k}k, saving model"
                            self.logger.info(msg)
                            tqdm.write(msg)
                            self.best_reward = rewards_val
                            self.save(self.logger.ckpt_dir, tag=f"reward_{rewards_val:.2f}")

        self.train_env.close()
        self.val_env.close()

        self._onTrainingEnds()

    def optimize(self):
        self.policy.setTrainingMode(True)
        self._updateLearningRate(
            self.lr_schedule, [self.actor.optimizer, self.critic.optimizer, self.ent_coef_optimizer]
        )

        actor_losses, critic_losses = [], []
        ent_coef_losses, ent_coefs = [], []
        for gradient_steps in range(self.config.gradient_steps):
            replay_data: ReplayBufferSamples = self.replay_buffer.sample(self.config.batch_size)
            # query action by the current actor on past observations
            actions_pi, log_prob = self.actor.actionLogProb(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # optimize ent coef, first detach it from the graph
            ent_coef = torch.exp(self.log_ent_coef.detach())
            # originally it would be `ent_coef` instead of `log_ent_coef`
            # but log is used to provide stability (same as original optimization)
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            # record ent stats
            ent_coef_losses.append(ent_coef_loss.item())
            ent_coefs.append(ent_coef.item())
            # optimize temperature parameter
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

            # classic optimization of Q function with TD error
            with torch.no_grad():
                next_actions_pi, next_log_prob = self.actor.actionLogProb(replay_data.next_observations)
                next_q_values = torch.min(*self.critic_target(replay_data.next_observations, next_actions_pi))
                # add entropy term, which is the log_prob * ent_coef
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = (
                    replay_data.rewards * self.config.reward_scale
                    + (1 - replay_data.dones) * self.config.gamma * next_q_values
                )

            # query online critic
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q_value, target_q_values) for current_q_value in current_q_values
            )
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # optimize actor with deterministic policy gradient
            min_q_values = torch.min(*self.critic(replay_data.observations, actions_pi))
            actor_loss = (ent_coef * log_prob - min_q_values).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # target updates
            if gradient_steps % self.config.target_update_interval == 0:
                polyakUpdate(self.critic.parameters(), self.critic_target.parameters(), self.config.tau)

        # record training stats
        self.recorder.record("train/ent coef", np.mean(ent_coefs), self._num_timesteps)
        self.recorder.record("train/ent coef loss", np.mean(ent_coef_losses), self._num_timesteps)
        self.recorder.record("train/actor loss", np.mean(actor_losses), self._num_timesteps)
        self.recorder.record("train/critic loss", np.mean(critic_losses), self._num_timesteps)

    def explore(self):
        self.policy.setTrainingMode(False)

        n_steps = 0
        while n_steps < self.config.train_freq:

            buffer_actions = self._selectAction(self._last_obs, deterministic=False)

            # rescale and pass into environment
            env_actions = self._toEnvAction(buffer_actions)
            new_obs, rewards, dones, infos = self.train_env.step(env_actions)
            truncateds = np.array([info.get("TimeLimit.truncated", False) for info in infos], dtype=np.bool_)

            # dump logs on train rewards
            self._recordTrainingStats(rewards=rewards, dones=dones)

            buffer_obs = new_obs.copy()
            if np.any(dones):
                for i in np.where(dones)[0]:
                    buffer_obs[i] = infos[i]["terminal_observation"]

            self.replay_buffer.push(self._last_obs, buffer_obs, buffer_actions, rewards, dones, truncateds)

            # update training progress
            n_steps += self.env_dim
            self._num_timesteps += self.env_dim
            self._n_calls += 1
            self._updateProgressRemaining(self._num_timesteps, self.total_iterations)

            # update last obs
            self._last_obs = new_obs.copy()

        return n_steps

    def predict(self, obs, deterministic=True) -> np.ndarray:
        """
        Return predicted actions in environment scale
        """
        self.policy.setTrainingMode(False)
        obs = toTensor(obs, device=self.device)
        with torch.no_grad():
            # query actor to sample from the underlying distribution
            actions = self.policy(obs, deterministic).cpu().numpy().reshape(-1, self.action_dim)
            actions = self._toEnvAction(actions)
            return actions

    def evaluate(self, env, eval_num, seed=None, action_scale=1.0):
        self.policy.setTrainingMode(False)
        env_dim = env.num_envs
        total_reward = 0.0
        for i in range(eval_num):
            base_seed = 0 if seed is None else seed
            env.seed(seed=base_seed + i * env_dim)
            obs = env.reset()
            dones_record = np.zeros(env_dim, dtype=np.bool_)
            while not dones_record.all():
                actions = self.predict(obs, True)
                obs, rewards, dones, _ = env.step(actions * action_scale)

                total_reward += (~dones_record * rewards).sum()
                dones_record |= dones

        return round(total_reward.item() / (env_dim * eval_num), 2)

    def setTrainingMode(self, train: bool = True):
        self.policy.setTrainingMode(train)

    def _selectAction(self, obs, deterministic) -> np.ndarray:
        """
        Sample normalized actions in training phase considering `learning_starts`
        Parameters:
            obs: shape (env_dim, obs_dim)
        Returns:
            action: action in (-1, 1) with shape (env_dim, action_dim)
        """
        if self._num_timesteps < self.config.train_start_iterations:
            # warm-up phase, action shape (env_dim, action_dim)
            action = np.random.uniform(-1, 1, size=(self.env_dim, self.action_dim))
        else:
            action = self.predict(obs, deterministic)
            # normalize back to (-1, 1)
            action = self._reverseEnvAction(action)
        return action

    def _toEnvAction(self, action: np.ndarray):
        """
        Scale actions from [-1, 1] to [low, high]
        Note:
            action is of shape (env_dim, *action_shape)
            self.action_low is of shape (env_dim, *action_shape)
        """
        return self.action_low + 0.5 * (self.action_high - self.action_low) * (action + 1.0)

    def _reverseEnvAction(self, action: np.ndarray):
        """
        Inverse transform of `_toEnvAction`, scale actions from [low, high] to [-1, 1]
        Note:
            action is of shape (env_dim, *action_shape)
            self.action_low is of shape (env_dim, *action_shape)
        """
        return 2.0 * (action - self.action_low) / (self.action_high - self.action_low) - 1.0

    def save(self, file_dir: str, tag: str) -> None:
        import os

        file_path = os.path.abspath(os.path.join(file_dir, f"./{str(self)}_{tag}.pth"))
        torch.save(self.policy.state_dict(), file_path)

    def load(self, file_path: str) -> None:
        self.policy.load_state_dict(torch.load(file_path, map_location=self.device))

    def __str__(self):
        return "SAC"
