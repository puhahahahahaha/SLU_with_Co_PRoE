import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from layers.utils import LayerNorm


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, attention_dropout=0.1):
        super(SelfAttention, self).__init__()

        self.num_attention_heads = 12
        self.attention_head_size = int(hidden_size / self.num_attention_heads)

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.out_size = out_size
        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.out_size)
        self.dropout = nn.Dropout(attention_dropout)

    def transpose_for_scores(self, x):
        last_dim = int(x.size()[-1] / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, last_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def roEmb(self, x, sin, cos):
        # x: batch_size * seq_len * dim
        # x: batch_size * num_head * seq_len * dim
        # x1: [0, 2, 4, 6, ...]
        # x2: [1, 3, 5, 7, ...]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        self.sin = sin
        self.cos = cos
        # 如果是旋转query key的话，下面这个直接cat就行，因为要进行矩阵乘法，最终会在这个维度求和。（只要保持query和key的最后一个dim的每一个位置对应上就可以）
        # torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        # 如果是旋转value的话，下面这个stack后再flatten才可以，因为训练好的模型最后一个dim是两两之间交替的。
        # batch_size * seq_len * dim
        return torch.stack([x1 * self.cos - x2 * self.sin, x2 * self.cos + x1 * self.sin], dim=-1).flatten(-2, -1)

    def forward(self, hidden, mask):
        # mask:     batch_size * seq_len
        # mask:     batch_size * 1 * 1 * seq_len
        # intent:   batch_size * seq_len * D
        # slot:     batch_size * seq_len * D
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        # 给所有不参与计算的值分配一个极大的 mask 值，以至于最后的 softmax 无限的接近于 0 ！！！
        attention_mask = (1.0 - extended_attention_mask) * -10000.0

        mixed_query_layer = self.query(hidden)
        mixed_key_layer = self.key(hidden)
        mixed_value_layer = self.value(hidden)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        sin, cos = self.RoE_QAndK(80).chunk(2, dim=-1)

        query_layer = self.roEmb(query_layer, sin, cos)
        key_layer = self.roEmb(key_layer, sin, cos)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_prob = nn.Softmax(dim=-1)(attention_scores)

        attention_prob = self.dropout(attention_prob)

        context_layer = torch.matmul(attention_prob, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.out_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer
