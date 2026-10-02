import inspect
import numpy as np
from copy import deepcopy
from gymnasium import spaces
from abc import ABC, abstractmethod
from typing import Any, Optional, Iterable, Union

from .types import VecEnvIndices, VecEnvObs, VecEnvStepReturn


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.

    :param num_envs: Number of environments
    :param observation_space: Observation space
    :param action_space: Action space
    """

    def __init__(
        self,
        num_envs: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
    ):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        # store info returned by the reset method
        self.reset_infos: list[dict[str, Any]] = [{} for _ in range(num_envs)]
        # seeds to be used in the next call to env.reset()
        self._seeds: list[Optional[int]] = [None for _ in range(num_envs)]
        # options to be used in the next call to env.reset()
        self._options: list[dict[str, Any]] = [{} for _ in range(num_envs)]

    def _reset_seeds(self) -> None:
        """
        Reset the seeds that are going to be used at the next reset.
        """
        self._seeds = [None for _ in range(self.num_envs)]

    def _reset_options(self) -> None:
        """
        Reset the options that are going to be used at the next reset.
        """
        self._options = [{} for _ in range(self.num_envs)]

    @abstractmethod
    def reset(self) -> VecEnvObs:
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.

        :return: observation
        """
        raise NotImplementedError()

    @abstractmethod
    def step_async(self, actions: np.ndarray) -> None:
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        raise NotImplementedError()

    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        """
        Wait for the step taken with step_async().

        :return: observation, reward, done, information
        """
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """
        Clean up the environment's resources.
        """
        raise NotImplementedError()

    def has_attr(self, attr_name: str) -> bool:
        """
        Check if an attribute exists for a vectorized environment.

        :param attr_name: The name of the attribute to check
        :return: True if 'attr_name' exists in all environments
        """
        # Default implementation, will not work with things that cannot be pickled:
        # https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/49
        try:
            self.get_attr(attr_name)
            return True
        except AttributeError:
            return False

    @abstractmethod
    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        """
        Return attribute from vectorized environment.

        :param attr_name: The name of the attribute whose value to return
        :param indices: Indices of envs to get attribute from
        :return: List of values of 'attr_name' in all environments
        """
        raise NotImplementedError()

    @abstractmethod
    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """
        Set attribute inside vectorized environments.

        :param attr_name: The name of attribute to assign new value
        :param value: Value to assign to `attr_name`
        :param indices: Indices of envs to assign value
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        """
        Call instance methods of vectorized environments.

        :param method_name: The name of the environment method to invoke.
        :param indices: Indices of envs whose method to call
        :param method_args: Any positional arguments to provide in the call
        :param method_kwargs: Any keyword arguments to provide in the call
        :return: List of items returned by the environment's method call
        """
        raise NotImplementedError()

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        """
        Step the environments with the given action

        :param actions: the action
        :return: observation, reward, done, information
        """
        self.step_async(actions)
        return self.step_wait()

    def get_images(self):
        """
        Return RGB images from each environment when available
        """
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None):
        """
        Sets the random seeds for all environments, based on a given seed.
        Each individual environment will still get its own seed, by incrementing the given seed.
        WARNING: since gym 0.26, those seeds will only be passed to the environment
        at the next reset.

        :param seed: The random seed. May be None for completely random seeding.
        :return: Returns a list containing the seeds for each individual env.
            Note that all list elements may be None, if the env does not return anything when being seeded.
        """
        if seed is None:
            # To ensure that subprocesses have different seeds,
            # we still populate the seed variable when no argument is passed
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))

        self._seeds = [seed + idx for idx in range(self.num_envs)]
        return self._seeds

    def set_options(self, options: Optional[Union[list[dict], dict]] = None) -> None:
        """
        Set environment options for all environments.
        If a dict is passed instead of a list, the same options will be used for all environments.
        WARNING: Those options will only be passed to the environment at the next reset.

        :param options: A dictionary of environment options to pass to each environment at the next reset.
        """
        if options is None:
            options = {}
        # Use deepcopy to avoid side effects
        if isinstance(options, dict):
            self._options = deepcopy([options] * self.num_envs)
        else:
            self._options = deepcopy(options)

    def getattr_depth_check(self, name: str, already_found: bool) -> Optional[str]:
        """Check if an attribute reference is being hidden in a recursive call to __getattr__

        :param name: name of attribute to check for
        :param already_found: whether this attribute has already been found in a wrapper
        :return: name of module whose attribute is being shadowed, if any.
        """
        if hasattr(self, name) and already_found:
            return f"{type(self).__module__}.{type(self).__name__}"
        else:
            return None

    def _get_indices(self, indices: VecEnvIndices) -> Iterable[int]:
        """
        Convert a flexibly-typed reference to environment indices to an implied list of indices.

        :param indices: refers to indices of envs.
        :return: the implied list of indices.
        """
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices


class VecEnvWrapper(VecEnv):
    """
    Vectorized environment base class

    :param venv: the vectorized environment to wrap
    :param observation_space: the observation space (can be None to load from venv)
    :param action_space: the action space (can be None to load from venv)
    """

    def __init__(
        self,
        venv: VecEnv,
        observation_space: Optional[spaces.Space] = None,
        action_space: Optional[spaces.Space] = None,
    ):
        self.venv = venv

        super().__init__(
            num_envs=venv.num_envs,
            observation_space=observation_space or venv.observation_space,
            action_space=action_space or venv.action_space,
        )
        self.class_attributes = dict(inspect.getmembers(self.__class__))

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self) -> VecEnvObs:
        pass

    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        pass

    def seed(self, seed: Optional[int] = None):
        return self.venv.seed(seed)

    def set_options(self, options: Optional[Union[list[dict], dict]] = None) -> None:
        return self.venv.set_options(options)

    def close(self) -> None:
        return self.venv.close()

    def get_images(self):
        return self.venv.get_images()

    def has_attr(self, attr_name: str) -> bool:
        return self.venv.has_attr(attr_name)

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        return self.venv.set_attr(attr_name, value, indices)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def __getattr__(self, name: str) -> Any:
        """Find attribute from wrapped venv(s) if this wrapper does not have it.
        Useful for accessing attributes from venvs which are wrapped with multiple wrappers
        which have unique attributes of interest.
        """
        blocked_class = self.getattr_depth_check(name, already_found=False)
        if blocked_class is not None:
            own_class = f"{type(self).__module__}.{type(self).__name__}"
            error_str = (
                f"Error: Recursive attribute lookup for {name} from {own_class} is "
                f"ambiguous and hides attribute from {blocked_class}"
            )
            raise AttributeError(error_str)

        return self.getattr_recursive(name)

    def _get_all_attributes(self) -> dict[str, Any]:
        """Get all (inherited) instance and class attributes

        :return: all_attributes
        """
        all_attributes = self.__dict__.copy()
        all_attributes.update(self.class_attributes)
        return all_attributes

    def getattr_recursive(self, name: str) -> Any:
        """Recursively check wrappers to find attribute.

        :param name: name of attribute to look for
        :return: attribute
        """
        all_attributes = self._get_all_attributes()
        if name in all_attributes:  # attribute is present in this wrapper
            attr = getattr(self, name)
        elif hasattr(self.venv, "getattr_recursive"):
            # Attribute not present, child is wrapper. Call getattr_recursive rather than getattr
            # to avoid a duplicate call to getattr_depth_check.
            attr = self.venv.getattr_recursive(name)
        else:  # attribute not present, child is an unwrapped VecEnv
            attr = getattr(self.venv, name)

        return attr

    def getattr_depth_check(self, name: str, already_found: bool) -> Optional[str]:
        """See base class.

        :return: name of module whose attribute is being shadowed, if any.
        """
        all_attributes = self._get_all_attributes()
        if name in all_attributes and already_found:
            # this venv's attribute is being hidden because of a higher venv.
            shadowed_wrapper_class: Optional[str] = f"{type(self).__module__}.{type(self).__name__}"
        elif name in all_attributes and not already_found:
            # we have found the first reference to the attribute. Now check for duplicates.
            shadowed_wrapper_class = self.venv.getattr_depth_check(name, True)
        else:
            # this wrapper does not have the attribute. Keep searching.
            shadowed_wrapper_class = self.venv.getattr_depth_check(name, already_found)

        return shadowed_wrapper_class
