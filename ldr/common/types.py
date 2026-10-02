from torch import Tensor
from typing import NamedTuple


class ReplayBufferSamples(NamedTuple):
    observations: Tensor
    next_observations: Tensor
    actions: Tensor
    dones: Tensor
    rewards: Tensor


class TrajectoryReplayBufferSamples(NamedTuple):
    traj_obs_actions: Tensor
    observations: Tensor
    next_observations: Tensor
    actions: Tensor
    dones: Tensor
    rewards: Tensor


class TrajectoryReplayBufferWithDynamicsIDSamples(NamedTuple):
    traj_obs_actions: Tensor
    observations: Tensor
    next_observations: Tensor
    actions: Tensor
    dones: Tensor
    rewards: Tensor
    dynamics_ids: Tensor
