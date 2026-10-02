from torch import nn
from typing import Iterable


def buildMLP(layer_shape: Iterable[int], activation, output_activation):
    """
    Build MLP from list
    """
    layers = []
    for j in range(len(layer_shape) - 1):
        act = activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)
