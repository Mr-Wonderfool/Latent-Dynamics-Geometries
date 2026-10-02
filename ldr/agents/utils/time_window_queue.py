import numpy as np
from typing import Optional


class TimeWindowQueue:
    def __init__(self, env_dim, n_past, obs_dim, padding: Optional[float] = None):
        self.n_past = n_past
        self.padding = padding
        self.window = np.ones((env_dim, n_past, obs_dim), dtype=np.float32)
        if self.padding is not None:
            self.window *= self.padding

        self.ptr = 0
        # record entries to be reset
        self.dones: np.ndarray = np.ones((env_dim,), dtype=bool)

    def append(self, obs: np.ndarray):
        if self.dones.any():
            self._resetDoneEntries(obs)
        self.window[:, self.n_past - self.ptr - 1, :] = obs
        self.ptr = (self.ptr + 1) % self.n_past

    def get(self):
        """
        Buffer consists of [0, 1, ..., T-1] data, move recorded data to front,
        forming [t, t-1, ..., T-1, (padding) 0, 1, ...]
        """
        windowed_data = np.roll(self.window, shift=self.ptr, axis=1)
        return windowed_data

    def _resetDoneEntries(self, initial_obs):
        if self.padding is not None:
            self.window[self.dones] = self.padding
        else:
            # repeat the initial observation for all timesteps for done environments
            self.window[self.dones] = initial_obs[self.dones][:, None, :]
        self.dones &= False

    def resetDoneEntries(self, dones: np.ndarray):
        """
        Mark entries for reset. If padding is not `None`, queue is reset immediately,
        otherwise, wait till the next `append` call to reset the queue with initial input.
        """
        self.dones = dones.copy()
        if self.padding is not None:
            self.window[self.dones] = self.padding
            self.dones &= False
        # else wait for next append call
