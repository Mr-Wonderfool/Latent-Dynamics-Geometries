import torch
import torch.nn as nn
from typing import List

from ..models.utils import buildMLP
from ..models.modules.tcn import TCN


class AdaptationNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        history_len: int,
        embed_dim: int,
        kernel_size_list: List[int],
        stride_list: List[int],
    ):
        super().__init__()
        input_dim = obs_dim + action_dim
        embed_mlp_dims = [input_dim, embed_dim, embed_dim]
        self.embedding = buildMLP(layer_shape=embed_mlp_dims, activation=nn.ReLU, output_activation=nn.ReLU)
        self.n_past = history_len

        self.tcn = TCN(embed_dim=embed_dim, kernel_size_list=kernel_size_list, stride_list=stride_list)

        # calculate output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, history_len, input_dim)
            # (batch, seq, feature) -> (batch, feature, seq)
            embedded = self.embedding(dummy_input).permute(0, 2, 1)
            tcn_output = self.tcn(embedded)
            flattend_size = tcn_output.reshape(1, -1).size(1)

        self.latent_pred = nn.Linear(flattend_size, latent_dim)

    def forward(self, states_with_actions: torch.Tensor):
        # flip input sequence from [new, ..., old] to [old, ..., new]
        states_with_actions_chron = torch.flip(states_with_actions, dims=[1])
        # (batch, seq, embed_dim) -> (batch, embed_dim, seq)
        embedded = self.embedding(states_with_actions_chron).permute(0, 2, 1)
        tcn_out = self.tcn(embedded)
        flattened = tcn_out.reshape(tcn_out.size(0), -1)
        z_pred = self.latent_pred(flattened)

        return z_pred
