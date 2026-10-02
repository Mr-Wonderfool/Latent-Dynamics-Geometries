import torch
import numpy as np


class BaseBuffer:
    def __init__(self, capacity: int, env_dim: int, obs_dim: int, action_dim: int):
        self.capacity = max(capacity // env_dim, 1)
        self.env_dim = env_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.full = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def toTorch(self, array: np.ndarray, copy: bool = True):
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default
        """
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)

    @classmethod
    def swapAndFlatten(cls, arr: np.ndarray):
        """
        Original data [n_envs, n_steps, ...],
        convert shape to [n_steps, n_envs, ...] and flatten the axes
        (order preserving so samples from one environment are together)
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    @property
    def size(self):
        if self.full:
            return self.capacity
        return self.ptr

    def push(self, *args, **kwargs):
        raise NotImplementedError()

    def sample(self, batch_size: int):
        batch_inds = np.random.randint(0, self.size, size=batch_size)
        return self._getSamples(batch_inds)

    def _getSamples(self, batch_inds: np.ndarray):
        raise NotImplementedError()
