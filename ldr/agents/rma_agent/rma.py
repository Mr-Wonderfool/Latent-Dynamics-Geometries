import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn

from ..utils.utils import toTensor
from .policies import RMASACPolicy
from ..sac_agent.sac import SACAgent
from .adapt_module import AdaptationNet
from ..utils.time_window_queue import TimeWindowQueue
from .rma_adapt_buffer import RMAAdaptBufferSamples, RMAAdaptBuffer
from .rma_config import RMAConfig, RMAAdaptModuleConfig, RMACMAConfig


class RMAAgent(SACAgent):
    config: RMAConfig

    def __init__(self, env, config, logger, recorder, device):
        super().__init__(
            env=env, config=config, logger=logger, recorder=recorder, device=device, _policy_cls=RMASACPolicy
        )
        self.factor_dim = self.config.policy["feature_extractor"]["kwargs"]["factor_dim"]
        self.student = None

    def buildAdaptationModule(self, config: RMAAdaptModuleConfig) -> None:
        """
        Build RMA phase 2 adaptation network and assign as attribute
        """
        self.student = AdaptationNet(
            obs_dim=self.obs_dim - self.factor_dim,
            action_dim=self.action_dim,
            latent_dim=self.config.latent_dim,
            **config.adapt_net.policy_kwargs,
        ).to(self.device)

    def loadAdaptationModule(self, config: RMAAdaptModuleConfig, file_path: str):
        """
        Build the adaptation network and load pretrained weights
        """
        self.buildAdaptationModule(config)
        self.student.load_state_dict(torch.load(file_path, map_location=self.device))
        self.student.train(False)

    def adaptWithModule(self, config: RMAAdaptModuleConfig, save_tag: str = "module_adapt", seed: int = 42) -> str:
        """
        Returns:
            path to the saved weights for adaptation module.
        """
        self.train_env.seed(seed)
        # global variables
        _last_obs = self.train_env.reset()
        last_stacked_obs_action = None
        # create aliases
        factor_mlp = self.actor.feature_extractor
        latent_dim = self.config.latent_dim
        obs_dim = self.obs_dim - self.factor_dim
        optimizer_config = config.adapt_net.optimizer
        n_past = config.adapt_net.policy_kwargs["history_len"]

        self.buildAdaptationModule(config=config)
        factor_mlp.eval()

        adapt_net_optimizer_class = optimizer_config.optimizerFromName(optimizer_config.name)
        adapt_net_scheduler_kwargs = optimizer_config.scheduler
        adapt_net_lr_schedule = optimizer_config.schedulerFromName(
            scheduler_name=adapt_net_scheduler_kwargs.pop("name"), scheduler_kwargs=adapt_net_scheduler_kwargs
        )
        adapt_net_optimizer = adapt_net_optimizer_class(self.student.parameters(), lr=adapt_net_lr_schedule(1))
        loss_fn = nn.MSELoss()

        # buffer setup
        replay_buffer = RMAAdaptBuffer(
            capacity=config.buffer_capacity,
            env_dim=self.env_dim,
            obs_dim=obs_dim,
            action_dim=self.action_dim,
            latent_dim=latent_dim,
            n_past=n_past,
        )
        obs_queue = TimeWindowQueue(env_dim=self.env_dim, n_past=n_past + 1, obs_dim=obs_dim, padding=None)
        action_queue = TimeWindowQueue(env_dim=self.env_dim, n_past=n_past, obs_dim=self.action_dim, padding=0)

        def _selectAction(obs_with_factor, deterministic: bool = False):
            """
            Return latent predicted by student, teacher, and the policy action modulated by latent from student
            """
            nonlocal last_stacked_obs_action
            clean_obs = obs_with_factor[..., : -self.factor_dim]
            obs_queue.append(clean_obs)
            # (env_dim, n_past+1, obs_dim), containing current
            full_stacked_obs = obs_queue.get()
            stacked_past_actions = action_queue.get()
            # infer z with new adaptation network
            current_obs, last_stacked_obs_action = self._preparePolicyAndEncoderInput(
                full_history_obs=full_stacked_obs, history_actions=stacked_past_actions
            )

            stacked_obs_tensor = toTensor(last_stacked_obs_action, device=self.device)
            curr_obs_tensor = toTensor(current_obs, device=self.device)
            curr_obs_with_factor_tensor = toTensor(obs_with_factor, device=self.device)
            with torch.no_grad():
                # use pred z to modulate policy
                student_z = self.student(states_with_actions=stacked_obs_tensor)
                curr_obs_with_latent = torch.cat([curr_obs_tensor, student_z], dim=-1)
                actions = self._forwardActor(obs_with_latent=curr_obs_with_latent, deterministic=deterministic)
                actions = actions.cpu().numpy().reshape(-1, self.action_dim)
                # query teacher policy for ground-truth z
                teacher_z = factor_mlp(curr_obs_with_factor_tensor)[..., -latent_dim:]
                return teacher_z.cpu().numpy(), actions

        # training loop
        total_num_timesteps = 0
        with tqdm(
            total=config.total_timesteps,
            desc="Adaptation Phase",
            unit="step",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            while total_num_timesteps < config.total_timesteps:
                # collect transitions to fill the buffer
                self.setTrainingMode(False)
                n_collect_steps = 0
                while n_collect_steps < config.train_freq:
                    n_collect_steps += self.env_dim
                    teacher_z, actions = _selectAction(_last_obs, deterministic=False)
                    action_queue.append(actions)
                    # rescale and pass into environment
                    env_actions = self._toEnvAction(actions)
                    new_obs, _, dones, _ = self.train_env.step(env_actions)
                    if np.any(dones):
                        obs_queue.resetDoneEntries(dones)
                        action_queue.resetDoneEntries(dones)

                    replay_buffer.push(traj_obs_actions=last_stacked_obs_action, gt_latent=teacher_z)
                    _last_obs = new_obs.copy()

                total_num_timesteps += n_collect_steps
                pbar.update(n_collect_steps)

                if total_num_timesteps > config.train_start_iterations:
                    # gradient steps on buffer data
                    self.setTrainingMode(True)
                    losses = []
                    for _ in range(config.gradient_steps):
                        replay_data: RMAAdaptBufferSamples = replay_buffer.sample(batch_size=config.batch_size)
                        pred_z = self.student(replay_data.traj_obs_actions)
                        loss = loss_fn(pred_z, replay_data.gt_latent)
                        adapt_net_optimizer.zero_grad()
                        loss.backward()
                        adapt_net_optimizer.step()

                        losses.append(loss.item())

                    self.recorder.record("adapt/mse", np.mean(losses), total_num_timesteps)

        save_dir = os.path.join(self.logger.log_dir, save_tag)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "module_adapt.pth")
        torch.save(self.student.state_dict(), save_path)
        return save_path

    def evaluateWithModule(self, env, eval_num, seed=None, action_scale=1.0):
        assert self.student is not None
        self.student.train(False)
        self.policy.setTrainingMode(False)

        env_dim = env.num_envs
        total_reward = 0.0

        obs_queue = TimeWindowQueue(
            env_dim=env_dim, n_past=self.student.n_past + 1, obs_dim=self.obs_dim - self.factor_dim, padding=None
        )
        action_queue = TimeWindowQueue(env_dim=env_dim, n_past=self.student.n_past, obs_dim=self.action_dim, padding=0)
        for i in range(eval_num):
            base_seed = 0 if seed is None else seed
            env.seed(seed=base_seed + i * env_dim)
            obs = env.reset()
            dones_record = np.zeros(env_dim, dtype=np.bool_)
            while not dones_record.all():
                # ! input obs still contains env factor, explicitly slice it
                obs_queue.append(obs[:, : -self.factor_dim])
                full_stacked_obs = obs_queue.get()
                stacked_past_actions = action_queue.get()
                current_obs, history_obs_actions = self._preparePolicyAndEncoderInput(
                    full_history_obs=full_stacked_obs, history_actions=stacked_past_actions
                )
                stacked_obs_tensor = toTensor(history_obs_actions, device=self.device)
                curr_obs_tensor = toTensor(current_obs, device=self.device)
                with torch.no_grad():
                    student_z = self.student(states_with_actions=stacked_obs_tensor)
                    curr_obs_with_latent = torch.cat([curr_obs_tensor, student_z], dim=-1)
                    actions = self._forwardActor(obs_with_latent=curr_obs_with_latent, deterministic=True)
                    actions = actions.cpu().numpy().reshape(-1, self.action_dim)

                action_queue.append(actions)
                env_actions = self._toEnvAction(actions)

                obs, rewards, dones, _ = env.step(env_actions * action_scale)
                total_reward += (~dones_record * rewards).sum()
                dones_record |= dones

            obs_queue.resetDoneEntries(dones_record)
            action_queue.resetDoneEntries(dones_record)

        return round(total_reward.item() / (env_dim * eval_num), 2)

    def adaptWithCMA(self, config: RMACMAConfig, seed: int = 42, action_scale: float = 1.0):
        """
        Returns:
            The searched scale for dynamics parameters.
        """
        try:
            import cma
        except ImportError:
            raise ValueError("Running RMA-CMA needs the `cma` package which is not installed!")
        assert 1 == self.env_dim, "CMA-ES requires exactly 1 environment for fine-grained control!"
        # record num samples
        num_samples = 0

        def fitness(candidate_z, n_episodes: int) -> float:
            nonlocal num_samples
            total_reward = 0.0

            self.train_env.seed(seed)
            obs = self.train_env.reset()
            for _ in range(n_episodes):
                done = False
                episode_reward = 0.0
                while not done:
                    obs_with_latent = np.concatenate([obs[:, : -self.factor_dim], candidate_z[None]], axis=1)
                    with torch.no_grad():
                        obs_with_latent = toTensor(obs_with_latent, device=self.device)
                        actions = self._forwardActor(obs_with_latent, deterministic=True)
                        actions = actions.cpu().numpy().reshape(-1, self.action_dim)

                    env_actions = self._toEnvAction(actions)

                    obs, rewards, dones, _ = self.train_env.step(env_actions * action_scale)
                    num_samples += 1

                    done = dones.item()
                    episode_reward += rewards.item()

                total_reward += episode_reward
            return total_reward / n_episodes

        x0 = np.ones(self.config.latent_dim)
        bounds = [config.lower_bound, config.upper_bound]
        es = cma.CMAEvolutionStrategy(x0, config.init_std, options={"bounds": bounds, "verbose": -1, "seed": seed})
        best_fitness = -np.inf
        best_params = x0.copy()

        for generation in tqdm(range(config.cma_max_iters), desc="CMA-ES Iteration"):
            candidates = es.ask()
            fitness_scores = []
            for candidate in candidates:
                reward = fitness(candidate_z=candidate, n_episodes=config.rollout_episodes)
                fitness_scores.append(-reward)
            es.tell(candidates, fitness_scores)

            # record current rollout reward
            real_rewards = [-f for f in fitness_scores]
            current_best = max(real_rewards)
            mean_reward = np.mean(real_rewards)

            if current_best > best_fitness:
                best_fitness = current_best
                best_idx = np.argmax(real_rewards)
                best_params = candidates[best_idx]

            self.recorder.record("cma_adapt/mean_reward", mean_reward, generation)
            self.recorder.record("cma_adapt/max_reward", current_best, generation)
            self.logger.info(f"CMA iteration {generation} has mean reward {mean_reward}, max reward: {current_best}")

            if es.stop():
                break

        self.logger.info(f"Total number of samples: {num_samples}\nIdentified z: {best_params}")

        return best_params

    def evaluateWithCMA(self, identified_z, env, eval_num, seed=None, action_scale=1.0):
        self.policy.setTrainingMode(False)
        env_dim = env.num_envs
        total_reward = 0.0
        for i in range(eval_num):
            base_seed = 0 if seed is None else seed
            env.seed(seed=base_seed + i * env_dim)
            obs = env.reset()
            dones_record = np.zeros(env_dim, dtype=np.bool_)
            while not dones_record.all():
                # slice factors
                obs_with_latent = np.concatenate([obs[:, : -self.factor_dim], identified_z[None]], axis=1)
                with torch.no_grad():
                    obs_with_latent = toTensor(obs_with_latent, device=self.device)
                    actions = self._forwardActor(obs_with_latent, deterministic=True)
                    actions = actions.cpu().numpy().reshape(-1, self.action_dim)

                env_actions = self._toEnvAction(actions)
                obs, rewards, dones, _ = env.step(env_actions * action_scale)

                total_reward += (~dones_record * rewards).sum()
                dones_record |= dones

        return round(total_reward.item() / (env_dim * eval_num), 2)

    def _forwardActor(self, obs_with_latent: torch.Tensor, deterministic: bool = False):
        latent_pi = self.actor.latent_pi(obs_with_latent)
        mean_actions = self.actor.mu(latent_pi)
        log_std = self.actor.log_std(latent_pi)
        log_std = torch.clamp(log_std, self.actor.log_std_min, self.actor.log_std_max)
        return self.actor.action_dist.actionFromParams(
            mean_actions=mean_actions, log_std=log_std, deterministic=deterministic
        )

    def _preparePolicyAndEncoderInput(self, full_history_obs: np.ndarray, history_actions: np.ndarray):
        current_obs = full_history_obs[:, 0]
        history_obs_actions = np.concatenate([full_history_obs[:, 1:], history_actions], axis=-1)
        return current_obs, history_obs_actions

    def __str__(self):
        return "RMA"
