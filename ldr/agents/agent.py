import os
import torch
import numpy as np
from typing import Union, List, Optional

from .agent_config import AgentConfig
from ldr.common.schedule import BaseSchedule
from ldr.common.utils import Logger, Recorder
from .utils.reward_monitor import RewardMonitor
from ldr.environments.vec_env.base_vec_env import VecEnv


class Agent:
    def __init__(self, env: VecEnv, config: AgentConfig, seed: int, logger: Logger, recorder: Recorder):
        """
        A base class for reinforcement learning agents.

        Parameters:
            lr_schedule (BaseSchedule): Schedule for dynamically adjusting the learning rate.
            seed (int): seed to apply for evaluation phase.
            logger (Logger): A logger for tracking events and metrics.
            recorder (Recorder): A recorder for tracking tensorboard type data.
        """
        self.config = config
        self.seed = seed
        self.logger = logger
        self.recorder = recorder
        # record training progress
        self.reward_monitor = RewardMonitor(
            num_envs=env.num_envs,
            total_timesteps=self.config.pipeline.train.total_iterations,
            expected_steps_per_env=env.envs[0].spec.max_episode_steps / 3,
            reward_dir=os.path.join(self.logger.log_dir, "monitor") if self.logger else None,
        )

        # initialize inner attribute according to environment
        self.env_dim = env.num_envs
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        # declare environments
        self.train_env = env
        self.val_env = None

        # current number of timesteps (multiple of env_dims)
        self._num_timesteps = 0
        self._current_progress_remaining = 1.0
        # number of calls for interactions (multiple of 1)
        self._n_calls = 0
        # storage of last observation
        self._last_obs = None

        # record training rewards
        self.train_rewards = np.zeros(self.env_dim)

    def _onTrainingStarts(self):
        """
        Set up initial variables at start of training, meanwhile reset the
        training environment.
        """
        assert self.val_env is not None, "Validation environment is not properly set yet!"

        self._num_timesteps = 0
        self._last_obs = self.train_env.reset()

    def _updateProgressRemaining(self, num_steps, total_steps):
        self._current_progress_remaining = 1.0 - float(num_steps) / float(total_steps)

    def _updateLearningRate(
        self, lr_schedule: BaseSchedule, optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer]
    ):
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).
        """
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule(self._current_progress_remaining)

    def _recordTrainingStats(self, rewards: np.ndarray, dones: np.ndarray):
        """
        Record episodic reward during training.
        Parameters:
            rewards (np.ndarray): rewards from environment steps, shape (env_dim, )
            dones (np.ndarray): done signal in environments, used to reset `train_rewards`,
                shape (env_dim, )
        """
        self.reward_monitor.recordReward(rewards, dones)
        # TODO: delete tmp recording of reward with tensorboard
        self.train_rewards += rewards
        if np.any(dones):
            self.recorder.record("train/train reward", np.mean(self.train_rewards[dones]), self._num_timesteps)
            self.train_rewards[dones] = 0.0

    def _onTrainingEnds(self):
        self.reward_monitor.writeToFile()

    def setValEnv(self, val_env: VecEnv):
        """
        Set validation and test environment to prepare for evaluation.
        Parameters:
            val_env (SimEnv): The simulation environment for validation.
        """
        self.val_env = val_env

    def explore(self):
        """
        Interact with the environment to collect experience.
        """
        raise NotImplementedError

    def train(self):
        """
        Train the agent by interacting with the environment, logging results, and optimizing the network.
        """
        raise NotImplementedError

    def evaluate(self, env: VecEnv, eval_num: int, seed: Optional[int] = None):
        """
        Evaluate the performance of an agent in a given environment over `env.num_envs * eval_num` episodes.

        Parameters:
            env (SimEnv): The evaluation environment where the agent will be tested.
            eval_nums (int): The number of evaluation episodes to run.
        """
        raise NotImplementedError

    def optimize(self):
        """
        Perform the optimization process for the agent's neural network.
        """
        raise NotImplementedError

    def predict(self, obs: Union[np.ndarray, torch.Tensor], deterministic: bool = True) -> np.ndarray:
        """
        Select an action based on the current observation during inference phase.

        Parameters:
            obs: The current observation of the environment.
            deterministic (bool): Whether to select actions deterministically (default: True).
        Returns:
            action: actions in original environment bounds
        """
        raise NotImplementedError

    def save(self, file_dir: str) -> None:
        """
        Save the agent's model and related data to a file.

        Parameters:
            file_dir (str): The directory where the model file will be saved.
        """
        raise NotImplementedError

    def load(self, file_dir: str) -> None:
        """
        Load the agent's model and related data from a file.

        Parameters:
            file_dir (str): The directory containing the saved model.
        """
        raise NotImplementedError
