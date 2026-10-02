import os
import sys
import time
import signal
import numpy as np
import pandas as pd


class RewardMonitor:
    """
    Record rewards for vectorized environments.
    """

    OUT_FILE = "monitor.csv"

    def __init__(self, num_envs: int, total_timesteps: int, expected_steps_per_env: int, reward_dir: str):
        self.t_start = time.time()
        self.num_envs = num_envs
        self.global_step_cnt = 0

        expected_reward_len_per_env = int(total_timesteps / num_envs / expected_steps_per_env)

        self.rewards_arr = np.empty((num_envs, expected_reward_len_per_env), dtype=np.float32)
        self.episode_len_arr = np.empty((num_envs, expected_reward_len_per_env), dtype=np.int32)
        self.time_arr = np.empty((num_envs, expected_reward_len_per_env), dtype=np.float32)
        self.global_steps_arr = np.empty((num_envs, expected_reward_len_per_env), dtype=np.int64)

        # internal variables
        self.size_ptr = np.zeros(num_envs, dtype=np.int32)
        self.per_env_stage_reward = np.zeros(num_envs, dtype=np.float32)
        self.per_env_step_counter = np.zeros(num_envs, dtype=np.int32)

        # file handler
        self.reward_dir = reward_dir
        if reward_dir is not None:
            self._registerSigintHandler()

    def recordReward(self, rewards: np.ndarray, dones: np.ndarray):
        self.per_env_stage_reward += rewards
        self.per_env_step_counter += 1
        self.global_step_cnt += self.num_envs

        if np.any(dones):
            curr_pointers = self.size_ptr[dones]
            self.rewards_arr[dones, curr_pointers] = self.per_env_stage_reward[dones]
            self.episode_len_arr[dones, curr_pointers] = self.per_env_step_counter[dones]
            self.time_arr[dones, curr_pointers] = time.time() - self.t_start
            self.global_steps_arr[dones, curr_pointers] = self.global_step_cnt

            self.per_env_stage_reward[dones] = 0
            self.per_env_step_counter[dones] = 0
            self.size_ptr[dones] += 1

        if np.any(self.size_ptr == self.rewards_arr.shape[1]):
            self._reAllocate()

    def writeToFile(self):
        os.makedirs(self.reward_dir, exist_ok=True)

        all_data = []
        for i in range(self.rewards_arr.shape[0]):
            end_index = self.size_ptr[i]
            env_data = pd.DataFrame(
                {
                    "reward": self.rewards_arr[i, :end_index],
                    "episode_length": self.episode_len_arr[i, :end_index],
                    "time": self.time_arr[i, :end_index],
                    "global_step": self.global_steps_arr[i, :end_index],
                }
            )
            env_data["episode_index"] = env_data.index
            all_data.append(env_data)

            env_data.to_csv(os.path.join(self.reward_dir, f"{i}.csv"), index=False)

        df_merged = pd.concat(all_data, ignore_index=True)
        # sort by recording time
        df_final = df_merged.sort_values(by=["global_step"]).reset_index(drop=True)
        file_path = os.path.join(self.reward_dir, self.OUT_FILE)
        df_final.to_csv(file_path, index=False)

    def _reAllocate(self):
        curr_size = self.rewards_arr.shape[1]
        new_size = (self.rewards_arr.shape[0], curr_size * 2)

        def resizeAndCopy(old_arr):
            new_arr = np.empty(new_size, dtype=old_arr.dtype)
            new_arr[:, :curr_size] = old_arr
            return new_arr

        self.rewards_arr = resizeAndCopy(self.rewards_arr)
        self.episode_len_arr = resizeAndCopy(self.episode_len_arr)
        self.time_arr = resizeAndCopy(self.time_arr)
        self.global_steps_arr = resizeAndCopy(self.global_steps_arr)

    def _sigintHandler(self, sig, frame):
        self.writeToFile()
        sys.exit(0)

    def _registerSigintHandler(self):
        signal.signal(signal.SIGINT, self._sigintHandler)
