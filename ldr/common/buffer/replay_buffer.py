import numpy as np

from .base_buffer import BaseBuffer
from ..types import ReplayBufferSamples


class ReplayBuffer(BaseBuffer):
    def __init__(self, env_dim: int, obs_dim: int, action_dim: int, capacity: int):
        super().__init__(capacity=capacity, env_dim=env_dim, obs_dim=obs_dim, action_dim=action_dim)
        self.observations = np.zeros((self.capacity, env_dim, obs_dim), dtype=np.float32)
        # if optimizing memory usage, next obs can be stored within obs
        self.next_observations = np.zeros((self.capacity, env_dim, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, env_dim, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, env_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, env_dim), dtype=np.float32)
        self.timeouts = np.zeros((self.capacity, env_dim), dtype=np.float32)

    def push(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        timeout: np.ndarray,
    ):
        input_action = action.reshape(self.env_dim, self.action_dim)
        # copy to avoid modification by reference
        self.observations[self.ptr] = np.array(obs)
        self.next_observations[self.ptr] = np.array(next_obs)
        self.actions[self.ptr] = np.array(input_action)
        self.rewards[self.ptr] = np.array(reward)
        self.dones[self.ptr] = np.array(done)
        self.timeouts[self.ptr] = np.array(timeout)

        self.ptr += 1
        if self.ptr == self.capacity:
            self.full = True
            self.ptr = 0

    def _getSamples(self, batch_inds: np.ndarray):
        env_indices = np.random.randint(0, high=self.env_dim, size=(len(batch_inds),))
        data = (
            self.observations[batch_inds, env_indices],
            self.next_observations[batch_inds, env_indices],
            self.actions[batch_inds, env_indices],
            # only use dones that are not due to timeouts
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.toTorch, data)))

    def save(self, file_path: str, **init_kwargs):
        curr_buffer_size = self.size
        # check for attributes to save
        _arrays = dict()
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                _arrays[key] = value[:curr_buffer_size]
        # slice only valid data, so saved buffer is full
        init_kwargs.update(
            env_dim=self.env_dim,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            capacity=curr_buffer_size * self.env_dim,
        )
        np.savez_compressed(
            file_path,
            ptr=curr_buffer_size,
            full=True,
            meta=np.array([init_kwargs], dtype=object),
            **_arrays,
        )

    @classmethod
    def load(cls, file_path: str) -> "ReplayBuffer":
        saved_buffer = np.load(file_path, allow_pickle=True)
        instance = cls(**saved_buffer.pop("meta")[0])
        # ! load ptr and full to use in `sample`
        instance.ptr = int(saved_buffer.pop("ptr").item())
        instance.full = bool(saved_buffer.pop("full").item())
        # check for shape match and load arrays
        for key, value in saved_buffer.items():
            instance_value = instance.__dict__[key]
            assert (
                instance_value.shape == value.shape
            ), f"Attribute {key} should have shape {instance_value.shape}, but loaded shape is {value.shape}"
            instance_value = value
        return instance
