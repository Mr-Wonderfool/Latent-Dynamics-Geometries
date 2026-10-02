from numpy import ndarray
from typing import Union, Iterable


# Define type aliases here to avoid circular import
# Used when we want to access one or more VecEnv
VecEnvIndices = Union[None, int, Iterable[int]]
# VecEnvObs is what is returned by the reset() method
# it contains the observation for each env
VecEnvObs = Union[ndarray, dict[str, ndarray], tuple[ndarray, ...]]
# VecEnvStepReturn is what is returned by the step() method
# it contains the observation, reward, done, info for each env
VecEnvStepReturn = tuple[VecEnvObs, ndarray, ndarray, list[dict]]
