import torch
from torch import nn
from typing import Tuple, Union
from abc import ABC, abstractmethod


class Distribution(ABC):
    """Abstract base class for all distributions"""

    def __init__(self):
        super().__init__()
        self.distribution = None

    @abstractmethod
    def probaDistributionNet(self, *args, **kwargs) -> Union[nn.Module, Tuple[nn.Module, nn.Parameter]]:
        """Create distribution for actions along with std"""

    @abstractmethod
    def probaDistribution(self, *args, **kwargs):
        """Set parameters for the distribution

        :return: self
        """

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """Sample stochastic actions from the distribution"""

    @abstractmethod
    def logProb(self, x: torch.Tensor) -> torch.Tensor:
        """Log likelihood regarding the taken action x"""

    @abstractmethod
    def mode(self) -> torch.Tensor:
        """Return the most likely action under the current distribution"""

    def getActions(self, deterministic: bool = False):
        if deterministic:
            return self.mode()
        return self.sample()
