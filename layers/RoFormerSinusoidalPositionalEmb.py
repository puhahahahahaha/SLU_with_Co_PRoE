from typing import Optional
import torch
import torch.nn as nn
import numpy as np


class RoFormerSinusoidalPositionalEmb(nn.Embedding):

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    # @torch.no_grad()
    # def forward(self, seq_len: int, past_key_values_length: int = 0):
    #     positions = torch.arange(
    #         past_key_values_length,
    #         past_key_values_length + seq_len,
    #         dtype=torch.long,
    #         device=self.weight.device,
    #     )
    #     # a = super().forward(positions)
    #     return super().forward(positions)
#
# RoE_QAndK = RoFormerSinusoidalPositionalEmbedding(60, 128)
# sinusoidal_pos = RoE_QAndK(60, 0)[None, None, :, :].chunk(2, dim=-1)
# sin, cos = sinusoidal_pos
