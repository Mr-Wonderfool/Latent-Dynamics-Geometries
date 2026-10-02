import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from ..sac_agent.sac import SACAgent
from .conditional_vae import Encoder, Decoder
from .ldr_config import LDRConfig, LDRAdaptConfig
from ..utils.time_window_queue import TimeWindowQueue
from ..utils.utils import toTensor, KLDiagGaussians, polyakUpdate
from ldr.environments.wrappers.global_wrappers import GlobalDynamicsIDWrapper
from ldr.common.buffer.traj_replay_buffer_with_id import (
    TrajectoryReplayBufferWithDynamicsID,
    TrajectoryReplayBufferWithDynamicsIDSamples,
)


class LDRAgent(SACAgent):
    config: LDRConfig

    def __init__(
        self,
        env,
        config: LDRConfig,
        logger,
        recorder,
        device,
    ):
        super().__init__(
            env=env,
            config=config,
            logger=logger,
            recorder=recorder,
            device=device,
            _init_setup_buffer=False,
        )
        kl_scheduler_kwargs = config.kl_schedule
        self.kl_schedule = config.optimizer.schedulerFromName(
            scheduler_name=kl_scheduler_kwargs.pop("name"), scheduler_kwargs=kl_scheduler_kwargs
        )
        self.past_obs_padding = None
        self.past_action_padding = 0
        # encoder decoder initialization
        self.encoder = Encoder(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            latent_dim=config.latent_dim,
            history_len=config.n_past,
            reg_mode=config.encoder_reg_mode,
            **config.encoder.policy_kwargs,
        ).to(self.device)
        encoder_optimizer_class = config.optimizer.optimizerFromName(config.encoder.optimizer.name)
        encoder_scheduler_kwargs = config.encoder.optimizer.scheduler
        self.encoder_lr_schedule = config.optimizer.schedulerFromName(
            scheduler_name=encoder_scheduler_kwargs.pop("name"), scheduler_kwargs=encoder_scheduler_kwargs
        )
        self.encoder_optimizer = encoder_optimizer_class(self.encoder.parameters(), lr=self.encoder_lr_schedule(1))

        self.decoder = Decoder(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            **config.decoder.policy_kwargs,
        ).to(self.device)
        decoder_optimizer_class = config.optimizer.optimizerFromName(config.decoder.optimizer.name)
        decoder_scheduler_kwargs = config.decoder.optimizer.scheduler
        self.decoder_lr_schedule = config.optimizer.schedulerFromName(
            scheduler_name=decoder_scheduler_kwargs.pop("name"), scheduler_kwargs=decoder_scheduler_kwargs
        )
        self.decoder_optimizer = decoder_optimizer_class(self.decoder.parameters(), lr=self.decoder_lr_schedule(1))
        # sequence length T+1, combine (0,....,T-1) with actions, pass T to policy
        self.obs_queue = TimeWindowQueue(
            env_dim=self.env_dim,
            n_past=config.n_past + 1,
            obs_dim=self.obs_dim,
            padding=self.past_obs_padding,
        )
        self.action_queue = TimeWindowQueue(
            env_dim=self.env_dim, n_past=config.n_past, obs_dim=self.action_dim, padding=self.past_action_padding
        )
        self.replay_buffer = TrajectoryReplayBufferWithDynamicsID(
            env_dim=self.env_dim,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            n_past=config.n_past,
            capacity=config.capacity,
        )

        # stacked obs action input to the encoder
        self._last_stacked_obs_action = None

        # record for training signal and apply latent dropout
        self.training = False
        self.dropout_prob = self.config.latent_dropout_prob

    def explore(self):
        self.training = True
        self.setTrainingMode(False)

        n_steps = 0
        while n_steps < self.config.train_freq:
            buffer_actions = self._selectAction(self._last_obs, deterministic=False)
            # store normalized action (with noise) into history buffer
            self.action_queue.append(buffer_actions)

            # rescale and pass into environment
            env_actions = self._toEnvAction(buffer_actions)
            new_obs, rewards, dones, infos = self.train_env.step(env_actions)
            truncateds = np.array([infos[i]["TimeLimit.truncated"] for i in range(self.env_dim)], dtype=np.bool_)

            # dump logs on train rewards
            self._recordTrainingStats(rewards=rewards, dones=dones)

            buffer_obs = new_obs.copy()
            if np.any(dones):
                for i in np.where(dones)[0]:
                    buffer_obs[i] = infos[i]["terminal_observation"]
                self.obs_queue.resetDoneEntries(dones)
                self.action_queue.resetDoneEntries(dones)

            curr_dynamics_ids = self._extractDynamicsIDFromInfos(infos)
            # current encoder input (batch, till t-1, (obs_dim+action_dim) * n_past)
            self.replay_buffer.push(
                traj_obs_actions=self._last_stacked_obs_action,
                curr_obs=self._last_obs,
                next_obs=buffer_obs,
                curr_action=buffer_actions,
                reward=rewards,
                done=dones,
                timeout=truncateds,
                dynamics_ids=curr_dynamics_ids,
            )

            # update training progress
            n_steps += self.env_dim
            self._num_timesteps += self.env_dim
            self._n_calls += 1
            self._updateProgressRemaining(self._num_timesteps, self.total_iterations)

            # update last obs
            self._last_obs = new_obs.copy()

        return n_steps

    def optimize(self):
        self.setTrainingMode(True)
        self._updateLearningRate(
            self.lr_schedule,
            [
                self.actor.optimizer,
                self.critic.optimizer,
                self.ent_coef_optimizer,
            ],
        )
        self._updateLearningRate(self.encoder_lr_schedule, [self.encoder_optimizer])
        self._updateLearningRate(self.decoder_lr_schedule, [self.decoder_optimizer])
        # KL annealing
        curr_beta = self.kl_schedule(self._current_progress_remaining)
        self.recorder.record("train/KL weight", curr_beta, self._num_timesteps)

        encoder_kl_losses, decoder_reconstruction_losses = [], []
        actor_losses, critic_losses = [], []
        ent_coef_losses, ent_coefs = [], []
        infonce_losses = []
        num_positives = []
        non_zero_percentages, max_num_positives, corres_dynamics_ids = [], [], []
        num_unique_ids_within_batch_record = []
        for gradient_steps in range(self.config.gradient_steps):
            replay_data: TrajectoryReplayBufferWithDynamicsIDSamples = self.replay_buffer.sample(
                self.config.batch_size, self.config.min_positives
            )
            # query encoder z with history
            q_z = self.encoder(states_with_actions=replay_data.traj_obs_actions)
            z = q_z.rsample()

            # apply latent dropout (for actor only), use mask to stop gradient flow
            mask = torch.bernoulli(torch.full((z.shape[0], 1), 1.0 - self.dropout_prob, device=z.device))
            z_rl = z * mask

            # add latent to observations
            obs_with_z = torch.cat([replay_data.observations, z_rl], dim=-1)
            # ! here we made an implicit assumption that z for the two observations are the same
            # ! while this assumption generally hold true, this does not enforce consistency
            next_obs_with_z = torch.cat([replay_data.next_observations, z_rl], dim=-1)

            """ Representation Learning Objective """
            dynamics_log_prob = self.decoder.logProb(
                observations=replay_data.observations,
                next_observations=replay_data.next_observations,
                actions=replay_data.actions,
                latent_vecs=z,
            )
            dynamics_loss = -dynamics_log_prob.mean()

            # KL div for encoder loss
            kl_div = KLDiagGaussians(
                ref_mean=q_z.loc, ref_std=q_z.scale, curr_mean=torch.zeros_like(z), curr_std=torch.ones_like(z)
            )
            kl_loss = kl_div.mean()

            vae_loss = dynamics_loss + curr_beta * kl_loss

            """ SAC Optimization """
            ent_coef = torch.exp(self.log_ent_coef.detach())

            # classic optimization of Q function with TD error
            with torch.no_grad():
                next_actions_pi, next_log_prob = self.actor.actionLogProb(next_obs_with_z)
                next_q_values = torch.min(*self.critic_target(next_obs_with_z, next_actions_pi))
                # add entropy term, which is the log_prob * ent_coef
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = (
                    replay_data.rewards * self.config.reward_scale
                    + (1 - replay_data.dones) * self.config.gamma * next_q_values
                )

            # query online critic
            current_q_values = self.critic(obs_with_z, replay_data.actions)
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q_value, target_q_values) for current_q_value in current_q_values
            )

            # optimize critic ahead of actor
            self.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic.optimizer.step()

            # actor loss with deterministic policy gradient
            actions_pi, log_prob = self.actor.actionLogProb(obs_with_z)
            log_prob = log_prob.reshape(-1, 1)
            # originally it would be `ent_coef` instead of `log_ent_coef`
            # but log is used to provide stability (same as original optimization)
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()

            min_q_values = torch.min(*self.critic(obs_with_z, actions_pi))
            actor_loss = (ent_coef * log_prob - min_q_values).mean()

            total_loss = actor_loss + self.config.vae_loss_weight * vae_loss

            if self._num_timesteps >= self.config.contrastive_start_iterations:
                (
                    infonce_loss,
                    mean_positives,
                    non_zero_percentage,
                    max_positives,
                    corres_dynamics_id,
                    num_unique_ids_within_batch,
                ) = self._computeContrastiveLoss(z=z, ids=replay_data.dynamics_ids)
                total_loss = total_loss + self.config.contrastive_loss_weight * infonce_loss
                num_positives.append(mean_positives.item())
                infonce_losses.append(infonce_loss.item())

                num_unique_ids_within_batch_record.append(num_unique_ids_within_batch)

                # recording number of positives
                non_zero_percentages.append(non_zero_percentage.item())
                max_num_positives.append(max_positives.item())
                corres_dynamics_ids.append(corres_dynamics_id.item())

            self.actor.optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            total_loss.backward()
            self.actor.optimizer.step()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            # optimize temperature parameter
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

            # target updates
            if gradient_steps % self.config.target_update_interval == 0:
                polyakUpdate(self.critic.parameters(), self.critic_target.parameters(), self.config.tau)

            """ Record Training Stats """
            encoder_kl_losses.append(kl_loss.item())
            decoder_reconstruction_losses.append(dynamics_loss.item())
            critic_losses.append(critic_loss.item())
            actor_losses.append(actor_loss.item())
            ent_coef_losses.append(ent_coef_loss.item())
            ent_coefs.append(ent_coef.item())

        self.recorder.record("train/ent coef", np.mean(ent_coefs), self._num_timesteps)
        self.recorder.record("train/ent coef loss", np.mean(ent_coef_losses), self._num_timesteps)
        self.recorder.record("train/actor loss", np.mean(actor_losses), self._num_timesteps)
        self.recorder.record("train/critic loss", np.mean(critic_losses), self._num_timesteps)
        self.recorder.record("train/encoder kl loss", np.mean(encoder_kl_losses), self._num_timesteps)
        self.recorder.record("train/decoder loss", np.mean(decoder_reconstruction_losses), self._num_timesteps)
        # record metric for contrastive learning
        if infonce_losses:
            self.recorder.record("train/infonce loss", np.mean(infonce_losses), self._num_timesteps)
        if num_positives:
            self.recorder.record("train/mean positives", np.mean(num_positives), self._num_timesteps)
            self.recorder.record("train/max positives for gradient update", np.max(num_positives), self._num_timesteps)
        # record number of dynamics till now
        unique_ids = np.unique(self.replay_buffer.dynamics_ids)
        self.recorder.record("train/number of ids", len(unique_ids), self._num_timesteps)
        if non_zero_percentages:
            self.recorder.record("train/non zero percentage", np.mean(non_zero_percentages), self._num_timesteps)
        if max_num_positives:
            self.recorder.record("train/mean max positives with batch", np.mean(max_num_positives), self._num_timesteps)
        if corres_dynamics_ids:
            bin_counts = np.bincount(corres_dynamics_ids)
            self.recorder.record("train/corresponding id", np.argmax(bin_counts), self._num_timesteps)

        if num_unique_ids_within_batch_record:
            self.recorder.record(
                "train/num unique ids within batch", np.mean(num_unique_ids_within_batch_record), self._num_timesteps
            )

    def _selectAction(self, obs, deterministic):
        self.obs_queue.append(obs)
        # (env_dim, n_past+1, obs_dim), containing current
        full_stacked_obs = self.obs_queue.get()
        stacked_past_actions = self.action_queue.get()
        current_obs, self._last_stacked_obs_action = self._preparePolicyAndEncoderInput(
            full_history_obs=full_stacked_obs, history_actions=stacked_past_actions
        )

        return self._predict(
            curr_obs=current_obs, stacked_past_obs_actions=self._last_stacked_obs_action, deterministic=deterministic
        )

    def _predict(
        self, curr_obs: np.ndarray, stacked_past_obs_actions: np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        """
        Returns:
            normalized actions in [-1, 1]
        """
        stacked_obs_tensor = toTensor(stacked_past_obs_actions, device=self.device)
        curr_obs_tensor = toTensor(curr_obs, device=self.device)
        with torch.no_grad():
            latent_z = self.encoder(stacked_obs_tensor)
            # since dynamics remain fixed, use mean to avoid introducing sampling error
            z = latent_z.loc

            # apply latent dropout (according to the prior distribution)
            if self.training:
                mask = torch.bernoulli(torch.full((z.shape[0], 1), 1.0 - self.dropout_prob, device=z.device))
                z = z * mask

            curr_obs_with_latent = torch.cat([curr_obs_tensor, z], dim=-1)
            actions = self.policy(obs=curr_obs_with_latent, deterministic=deterministic)
            actions = actions.cpu().numpy().reshape(-1, self.action_dim)
            return actions

    def _preparePolicyAndEncoderInput(self, full_history_obs: np.ndarray, history_actions: np.ndarray):
        current_obs = full_history_obs[:, 0]
        history_obs_actions = np.concatenate([full_history_obs[:, 1:], history_actions], axis=-1)
        return current_obs, history_obs_actions

    def _computeContrastiveLoss(self, z: torch.Tensor, ids: torch.Tensor):
        """
        Compute supervised constrastive learning loss (InfoNCE).
        Intuitively, z under different dynamics ids are pushed away from each other,
        while z under same dynamics ids stay closer.
        """
        z_norm = F.normalize(z, dim=1)
        batch_size = z.shape[0]

        unique_ids_within_batch = torch.unique(ids)

        # masks for positive and negative pairs
        ids_col = ids.view(-1, 1)
        ids_row = ids.view(1, -1)

        # positive mask: same ID but not same elements
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        positive_mask = (ids_col == ids_row) & ~self_mask

        # loss for positive pair (i, p) is:
        # exp(logits[i,p]) / sum_{k!=i} exp(logits[k,p])
        logits = torch.matmul(z_norm, z_norm.T) / self.config.infonce_temp
        logits_masked_self = logits.masked_fill(self_mask, float("-inf"))
        log_denominator = torch.logsumexp(logits_masked_self, dim=1, keepdim=True)
        log_probs = logits - log_denominator
        # with multiple positives, loss for i is: (1 / |P_i|) sum_{p in P(i)}
        sum_positive_log_probs = (log_probs * positive_mask).sum(dim=1)
        num_positives = positive_mask.sum(dim=1)

        # tmp logging to check the change of positives
        mean_positives = num_positives.float().mean()
        # tmp logging on number of elements that are not zero
        non_zero_ele = torch.count_nonzero(num_positives)
        non_zero_percentage = non_zero_ele / batch_size
        max_idx = torch.argmax(num_positives)
        max_positives = num_positives[max_idx]
        corres_dynamics_id = ids[max_idx]

        # safeguard cases where there are no positives in one batch
        num_positives = torch.clamp(num_positives, min=1.0)

        loss_per_anchor = -sum_positive_log_probs / num_positives
        loss = loss_per_anchor.mean()

        return (
            loss,
            mean_positives,
            non_zero_percentage,
            max_positives,
            corres_dynamics_id,
            len(unique_ids_within_batch),
        )

    def _extractDynamicsIDFromInfos(self, infos: dict):
        dynamics_ids = [info[GlobalDynamicsIDWrapper.MANAGER_DYNAMICS_ID_STR] for info in infos]
        return np.array(dynamics_ids, dtype=np.int64)

    def evaluate(self, env, eval_num, seed=None, action_scale=1.0):
        self.training = False
        self.setTrainingMode(False)
        env_dim = env.num_envs
        total_reward = 0.0

        # initialize queue based on environment dimension
        obs_queue = TimeWindowQueue(
            env_dim=env_dim, n_past=self.config.n_past + 1, obs_dim=self.obs_dim, padding=self.past_obs_padding
        )
        action_queue = TimeWindowQueue(
            env_dim=env_dim, n_past=self.config.n_past, obs_dim=self.action_dim, padding=self.past_action_padding
        )

        for i in range(eval_num):
            base_seed = 0 if seed is None else seed
            env.seed(seed=base_seed + i * env_dim)
            obs = env.reset()
            dones_record = np.zeros(env_dim, dtype=np.bool_)
            while not dones_record.all():
                obs_queue.append(obs)
                full_stacked_obs = obs_queue.get()
                stacked_past_actions = action_queue.get()
                current_obs, history_obs_actions = self._preparePolicyAndEncoderInput(
                    full_history_obs=full_stacked_obs, history_actions=stacked_past_actions
                )
                actions = self._predict(
                    curr_obs=current_obs, stacked_past_obs_actions=history_obs_actions, deterministic=True
                )
                action_queue.append(actions)
                env_actions = self._toEnvAction(actions)

                obs, rewards, dones, _ = env.step(env_actions * action_scale)
                total_reward += (~dones_record * rewards).sum()
                dones_record |= dones

            obs_queue.resetDoneEntries(dones_record)
            action_queue.resetDoneEntries(dones_record)

        return round(total_reward.item() / (env_dim * eval_num), 2)

    def save(self, file_dir: str, tag: str):
        model_dir = os.path.abspath(os.path.join(file_dir, f"./{str(self)}_{tag}"))
        os.makedirs(model_dir, exist_ok=True)
        policy_path = os.path.join(model_dir, "policy.pth")
        torch.save(self.policy.state_dict(), policy_path)
        encoder_path = os.path.join(model_dir, "encoder.pth")
        torch.save(self.encoder.state_dict(), encoder_path)
        decoder_path = os.path.join(model_dir, "decoder.pth")
        torch.save(self.decoder.state_dict(), decoder_path)

    def load(self, model_dir: str):
        policy_path = os.path.join(model_dir, "policy.pth")
        self.policy.load_state_dict(torch.load(policy_path, map_location=self.device))
        encoder_path = os.path.join(model_dir, "encoder.pth")
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        decoder_path = os.path.join(model_dir, "decoder.pth")
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))

    def setTrainingMode(self, train: bool = True):
        self.policy.setTrainingMode(train)
        self.encoder.train(train)
        self.decoder.train(train)

    def __str__(self):
        return "LDR"
