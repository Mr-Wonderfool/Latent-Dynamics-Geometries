import numpy as np

from .replay_buffer import ReplayBuffer
from ..types import TrajectoryReplayBufferSamples


class TrajectoryReplayBuffer(ReplayBuffer):
    def __init__(self, env_dim, obs_dim, action_dim: int, n_past: int, capacity):
        super().__init__(env_dim=env_dim, obs_dim=obs_dim, action_dim=action_dim, capacity=capacity)
        # add storage for observation history
        self.n_past = n_past
        self.traj_obs_actions = np.zeros(
            (self.capacity, env_dim, (obs_dim + self.action_dim) * n_past), dtype=np.float32
        )

    def push(
        self,
        traj_obs_actions: np.ndarray,
        curr_obs: np.ndarray,
        next_obs: np.ndarray,
        curr_action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        timeout: np.ndarray,
    ):
        self.traj_obs_actions[self.ptr] = traj_obs_actions.reshape(self.env_dim, -1)

        input_action = curr_action.reshape(self.env_dim, self.action_dim)
        self.observations[self.ptr] = np.array(curr_obs)
        self.next_observations[self.ptr] = np.array(next_obs)
        self.actions[self.ptr] = np.array(input_action)
        self.rewards[self.ptr] = np.array(reward)
        self.dones[self.ptr] = np.array(done)
        self.timeouts[self.ptr] = np.array(timeout)

        self.ptr += 1
        if self.ptr == self.capacity:
            self.full = True
            self.ptr = 0

    def _getSamples(self, batch_inds):
        env_indices = np.random.randint(0, high=self.env_dim, size=(len(batch_inds),))
        data = (
            self.traj_obs_actions[batch_inds, env_indices].reshape(-1, self.n_past, self.obs_dim + self.action_dim),
            self.observations[batch_inds, env_indices],
            self.next_observations[batch_inds, env_indices],
            self.actions[batch_inds, env_indices],
            # only use dones that are not due to timeouts
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
        )
        return TrajectoryReplayBufferSamples(*tuple(map(self.toTorch, data)))

    def save(self, file_path):
        return super().save(file_path, n_past=self.n_past)
