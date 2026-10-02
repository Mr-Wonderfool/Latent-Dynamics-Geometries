import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .palette import PlotPalette
from .base_visualizer import BaseVisualizer
from ldr.agents.utils.utils import toTensor
from ldr.environments import makeVectorizedEnv
from ldr.common.types import TrajectoryReplayBufferSamples
from ldr.common.buffer.traj_replay_buffer_with_id import TrajectoryReplayBufferWithDynamicsID


class StatePredictionVisualizer:
    def __init__(self, env_params: dict, agent_params: dict):
        assert agent_params["agent_name"] == "LDRAgent"
        BaseVisualizer.setTheme()

        env_config = BaseVisualizer.envConfigFromDict(env_params)
        env_config.num_envs = 1
        env_config.render = False
        self.env = makeVectorizedEnv(env_config)
        self.agent = BaseVisualizer.agentFromDict(
            agent_params=agent_params, venv=self.env, weight_path=agent_params["pipeline"]["test"]["model"]
        )

        # create aliases
        self.n_past = self.agent.config.n_past
        self.obs_dim = self.agent.obs_dim
        self.action_dim = self.agent.action_dim
        self.device = self.agent.device

    def rollout(self, rollout_timesteps: int = 1000, seed: int = 42):
        buffer = self.rolloutTimesteps(total_timesteps=rollout_timesteps, seed=seed)
        # seed for choosing state indices
        np.random.seed(seed)
        # use the first episode in the first environment
        env_idx = 0
        dones_episode = buffer.dones[:, env_idx]
        done_indices = np.where(dones_episode)[0]
        if done_indices.size > 0:
            end_idx = done_indices[0] + 1
        else:
            end_idx = buffer.capacity

        required_keys = ["traj_obs_actions", "observations", "next_observations", "actions"]
        buffer_samples = {}
        for key in required_keys:
            buffer_samples[key] = buffer.__dict__[key].swapaxes(0, 1)[env_idx : env_idx + 1, :end_idx]

        return TrajectoryReplayBufferSamples(**buffer_samples, rewards=None, dones=None)

    def visualizeStatePrediction(self, buffer_samples: TrajectoryReplayBufferSamples, figsize=(14, 8)):
        """
        Plots the true vs. predicted value of four random state variable based on the provided observations.
        """
        encoder = self.agent.encoder
        decoder = self.agent.decoder

        time_steps = np.arange(buffer_samples.observations.shape[1])

        with torch.no_grad():
            obs_tensor = toTensor(buffer_samples.observations, self.device)
            actions_tensor = toTensor(buffer_samples.actions, self.device)
            traj_tensor = toTensor(
                buffer_samples.traj_obs_actions.reshape(-1, self.n_past, self.obs_dim + self.action_dim),
                self.device,
            )

            q_z = encoder(states_with_actions=traj_tensor)
            z = q_z.loc

            # for normalized decoder
            pred_mean, pred_std = decoder.predictNextObs(observations=obs_tensor, actions=actions_tensor, latent_vecs=z)

            pred_mean = pred_mean[0].cpu().numpy()
            pred_std = pred_std[0].cpu().numpy()

        fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True)

        state_indices = np.random.choice(np.arange(self.env.observation_space.shape[0]), size=(4,), replace=False)
        for i in range(4):
            ax = axs[i // 2, i % 2]

            ax.plot(
                time_steps,
                buffer_samples.next_observations[0, :, state_indices[i]],
                label="ground truth",
                color=PlotPalette.red,
                linewidth=2,
            )
            ax.plot(
                time_steps,
                pred_mean[:, state_indices[i]],
                label="prediction mean",
                color=PlotPalette.blue,
                linestyle="--",
            )

            # Plot Predicted Std (sigma) as a filled area
            lower_bound = pred_mean[:, state_indices[i]] - pred_std[:, state_indices[i]]
            upper_bound = pred_mean[:, state_indices[i]] + pred_std[:, state_indices[i]]
            ax.fill_between(
                time_steps, lower_bound, upper_bound, color=PlotPalette.blue, alpha=0.2, label="prediction std"
            )

            ax.set_ylabel("Value")
            ax.grid(True, linestyle="--", alpha=0.6)
            if i // 2 == 1:
                ax.set_xlabel("Time Step (t)")

        axs[-1, -1].legend()
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        self.env.close()

        return fig, ax

    def rolloutTimesteps(self, total_timesteps: int, seed: int = 42):
        self.agent.setTrainingMode(False)
        # create aliases
        env = self.agent.train_env
        env.seed(seed)

        self.agent._last_obs = env.reset()
        # buffer for online optimization
        total_timesteps = total_timesteps // env.num_envs * env.num_envs
        buffer = TrajectoryReplayBufferWithDynamicsID(
            env_dim=env.num_envs,
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            n_past=self.agent.config.n_past,
            capacity=total_timesteps,
        )

        num_timesteps = 0
        with tqdm(
            total=total_timesteps,
            desc="Target Domain Rollout",
            unit="step",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            while num_timesteps < total_timesteps:
                buffer_actions = self.agent._selectAction(self.agent._last_obs, deterministic=True)
                self.agent.action_queue.append(buffer_actions)
                env_actions = self.agent._toEnvAction(buffer_actions)
                new_obs, rewards, dones, infos = self.agent.train_env.step(env_actions)
                truncateds = np.array([info.get("TimeLimit.truncated", False) for info in infos], dtype=np.bool_)

                buffer_obs = new_obs.copy()
                if np.any(dones):
                    for i in np.where(dones)[0]:
                        buffer_obs[i] = infos[i]["terminal_observation"]
                    self.agent.obs_queue.resetDoneEntries(dones)
                    self.agent.action_queue.resetDoneEntries(dones)

                # dynamics id is the same for all samples in target domain
                curr_dynamics_ids = np.zeros((env.num_envs,), dtype=np.int64)

                buffer.push(
                    traj_obs_actions=self.agent._last_stacked_obs_action,
                    curr_obs=self.agent._last_obs,
                    next_obs=buffer_obs,
                    curr_action=buffer_actions,
                    reward=rewards,
                    done=dones,
                    timeout=truncateds,
                    dynamics_ids=curr_dynamics_ids,
                )

                self.agent._last_obs = new_obs.copy()

                num_timesteps += env.num_envs
                pbar.update(env.num_envs)

        return buffer
