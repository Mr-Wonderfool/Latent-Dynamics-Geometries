import numpy as np
from torch import Tensor
from typing import NamedTuple

from ldr.common.buffer.base_buffer import BaseBuffer


class RMAAdaptBufferSamples(NamedTuple):
    traj_obs_actions: Tensor
    gt_latent: Tensor


class RMAAdaptBuffer(BaseBuffer):
    def __init__(self, capacity, env_dim, obs_dim, action_dim, latent_dim: int, n_past: int):
        super().__init__(capacity=capacity, env_dim=env_dim, obs_dim=obs_dim, action_dim=action_dim)
        self.n_past = n_past
        self.traj_obs_actions = np.zeros(
            (self.capacity, env_dim, (obs_dim + self.action_dim) * n_past), dtype=np.float32
        )
        self.gt_latent = np.zeros((self.capacity, env_dim, latent_dim), dtype=np.float32)

    def push(self, traj_obs_actions: np.ndarray, gt_latent: np.ndarray):
        self.traj_obs_actions[self.ptr] = traj_obs_actions.reshape(self.env_dim, -1)
        self.gt_latent[self.ptr] = gt_latent

        self.ptr += 1
        if self.ptr == self.capacity:
            self.full = True
            self.ptr = 0

    def _getSamples(self, batch_inds):
        env_indices = np.random.randint(0, high=self.env_dim, size=(len(batch_inds),))
        data = (
            self.traj_obs_actions[batch_inds, env_indices].reshape(-1, self.n_past, self.obs_dim + self.action_dim),
            self.gt_latent[batch_inds, env_indices],
        )
        return RMAAdaptBufferSamples(*tuple(map(self.toTorch, data)))
