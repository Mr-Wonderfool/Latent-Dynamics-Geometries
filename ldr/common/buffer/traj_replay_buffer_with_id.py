import numpy as np
from typing import Optional

from .traj_replay_buffer import TrajectoryReplayBuffer
from ..types import TrajectoryReplayBufferWithDynamicsIDSamples


class TrajectoryReplayBufferWithDynamicsID(TrajectoryReplayBuffer):
    def __init__(self, env_dim, obs_dim, action_dim, n_past: int, capacity):
        super().__init__(
            env_dim=env_dim,
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_past=n_past,
            capacity=capacity,
        )
        # add storage for dynamics id
        self.dynamics_ids = np.zeros((self.capacity, env_dim), dtype=np.int64)

    def push(
        self,
        traj_obs_actions: np.ndarray,
        curr_obs: np.ndarray,
        next_obs: np.ndarray,
        curr_action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        timeout: np.ndarray,
        dynamics_ids: np.ndarray,
    ):
        self.dynamics_ids[self.ptr] = np.array(dynamics_ids)
        self.traj_obs_actions[self.ptr] = traj_obs_actions.reshape(self.env_dim, -1)

        input_action = curr_action.reshape(self.env_dim, self.action_dim)
        self.observations[self.ptr] = np.array(curr_obs)
        self.next_observations[self.ptr] = np.array(next_obs)
        self.actions[self.ptr] = np.array(input_action)
        self.rewards[self.ptr] = np.array(reward)
        self.dones[self.ptr] = np.array(done)
        self.timeouts[self.ptr] = np.array(timeout)

        self.ptr += 1
        # if buffer is full, manually set ptr to avoid overwriting anchor id
        if self.ptr == self.capacity:
            self.full = True
            self.ptr = 0

    def sample(self, batch_size: int, min_positives: int = 1) -> TrajectoryReplayBufferWithDynamicsIDSamples:
        upper_bound = self.size

        if min_positives <= 1:
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
            return self._getSamples(batch_inds)

        # align batch size with min positives
        assert (
            batch_size % min_positives == 0
        ), f"Batch size {batch_size} is not divisible by min positives {min_positives}!"
        num_groups = batch_size // min_positives
        valid_ids = self.dynamics_ids[:upper_bound]
        unique_ids, counts = np.unique(valid_ids, return_counts=True)
        valid_ids_for_sampling = unique_ids[counts >= min_positives]
        if len(valid_ids_for_sampling) < num_groups:
            # fall back to original sampling method
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
            return self._getSamples(batch_inds)

        sampled_ids = np.random.choice(valid_ids_for_sampling, size=num_groups, replace=False)
        final_batch_inds = []
        final_env_inds = []
        for id in sampled_ids:
            # (N, 2) coords indicating position of data
            coords = np.argwhere(valid_ids == id)
            indices = np.random.randint(0, len(coords), size=min_positives)
            selected_coords = coords[indices]

            final_batch_inds.append(selected_coords[:, 0])
            final_env_inds.append(selected_coords[:, 1])

        batch_inds = np.concatenate(final_batch_inds)
        env_inds = np.concatenate(final_env_inds)
        return self._getSamples(batch_inds, env_inds)

    def _getSamples(self, batch_inds, env_inds: Optional[np.ndarray] = None):
        if env_inds is None:
            env_inds = np.random.randint(0, high=self.env_dim, size=(len(batch_inds),))
        data = (
            self.traj_obs_actions[batch_inds, env_inds].reshape(-1, self.n_past, self.obs_dim + self.action_dim),
            self.observations[batch_inds, env_inds],
            self.next_observations[batch_inds, env_inds],
            self.actions[batch_inds, env_inds],
            # only use dones that are not due to timeouts
            (self.dones[batch_inds, env_inds] * (1 - self.timeouts[batch_inds, env_inds])).reshape(-1, 1),
            self.rewards[batch_inds, env_inds].reshape(-1, 1),
            self.dynamics_ids[batch_inds, env_inds].reshape(-1, 1),
        )
        return TrajectoryReplayBufferWithDynamicsIDSamples(*tuple(map(self.toTorch, data)))
