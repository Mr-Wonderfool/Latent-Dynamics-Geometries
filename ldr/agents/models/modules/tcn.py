from torch import nn
from typing import List


class TCN(nn.Module):
    def __init__(self, embed_dim: int, kernel_size_list: List[int], stride_list: List[int]):
        super().__init__()

        num_layers = len(kernel_size_list)
        tcn_layers = []
        assert len(stride_list) == num_layers
        for i in range(num_layers):
            tcn_layers.append(nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size_list[i], stride=stride_list[i]))
            tcn_layers.append(nn.ReLU())
        self.tcn = nn.Sequential(*tcn_layers)

    def forward(self, seq):
        """
        Seq: [old, ..., new] with shape (batch, embed_dim, seq)
        """
        return self.tcn(seq)
