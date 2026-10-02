import torch
import numpy as np
from typing import Iterable


def toTensor(obs, device):
    """
    Convert numpy array to tensor on the given device
    """
    if isinstance(obs, torch.Tensor):
        return obs.float()
    if isinstance(obs, np.ndarray):
        return torch.as_tensor(obs, device=device).float()
    else:
        raise Exception("Unrecognized type for conversion")


def polyakUpdate(params: Iterable[torch.Tensor], target_params: Iterable[torch.Tensor], tau: float) -> None:
    """
    Perform a Polyak average update on `target_params` using `params`
    with tau serving as interpolation coefficient.
    Target_prime = (1 - tau) * target + tau * original
    """
    with torch.no_grad():
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def KLDiagGaussians(ref_mean, ref_std, curr_mean, curr_std):
    """
    Compute KL divergence KL(ref||curr) of two diagonal Gaussian distribution
    Parameters:
        ref_mean: mean of the reference entry
        ref_std: standard deviation of the reference entry
    """
    kl_div = torch.log(curr_std / ref_std) + (ref_std**2 + (ref_mean - curr_mean) ** 2) / (2.0 * curr_std**2) - 0.5
    return kl_div.sum(dim=-1)
