import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, intermediate_size, hidden_size, dropout):
        super(Intermediate, self).__init__()
        self.dense_in = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.ReLU()
        self.dense_out = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        hidden_states_new = self.dense_in(hidden_states)
        hidden_states_new = self.intermediate_act_fn(hidden_states_new)
        hidden_states_new = self.dense_out(hidden_states_new)
        hidden_states_new = self.dropout(hidden_states_new)
        hidden_states_new = self.LayerNorm(hidden_states_new + hidden_states)
        return hidden_states_new


class Intermediate_I_S(nn.Module):
    def __init__(self, intermediate_size, hidden_size, dropout=0.1):
        super(Intermediate_I_S, self).__init__()
        self.dense_in = nn.Linear(hidden_size * 6, intermediate_size)
        self.intermediate_act_fn = nn.ReLU()
        self.dense_out = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm_I = LayerNorm(hidden_size, eps=1e-12)
        self.LayerNorm_S = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states_I, hidden_states_S):
        hidden_states_in = torch.cat([hidden_states_I, hidden_states_S], dim=2)
        batch_size, max_length, hidden_size = hidden_states_in.size()
        h_pad = torch.zeros(batch_size, 1, hidden_size)
        if torch.cuda.is_available():
            h_pad = h_pad.cuda()
        h_left = torch.cat([h_pad, hidden_states_in[:, :max_length - 1, :]], dim=1)
        h_right = torch.cat([hidden_states_in[:, 1:, :], h_pad], dim=1)
        hidden_states_in = torch.cat([hidden_states_in, h_left, h_right], dim=2)
        hidden_states = self.dense_in(hidden_states_in)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states_I_NEW = self.LayerNorm_I(hidden_states + hidden_states_I)
        hidden_states_S_NEW = self.LayerNorm_S(hidden_states + hidden_states_S)
        return hidden_states_I_NEW, hidden_states_S_NEW


class I_S_Block(nn.Module):
    def __init__(self, intent_emb, slot_emb, hidden_size, attention_dropout, max_seq_len):
        super(I_S_Block, self).__init__()
        self.I_S_Attention = I_S_SelfAttention(hidden_size, 2 * hidden_size, hidden_size, attention_dropout,
                                               max_seq_len)
        self.I_Out = SelfOutput(hidden_size, attention_dropout)
        self.S_Out = SelfOutput(hidden_size, attention_dropout)
        self.I_Feed_forward = Intermediate(4 * hidden_size, hidden_size, attention_dropout)
        self.S_Feed_forward = Intermediate(4 * hidden_size, hidden_size, attention_dropout)

    def forward(self, H_intent_input, H_slot_input, mask):
        H_slot, H_intent = self.I_S_Attention(H_intent_input, H_slot_input, mask)
        H_slot = self.S_Out(H_slot, H_slot_input)
        H_intent = self.I_Out(H_intent, H_intent_input)
        H_slot = self.S_Feed_forward(H_slot)
        H_intent = self.I_Feed_forward(H_intent)

        return H_intent, H_slot


class Label_Attention(nn.Module):
    def __init__(self, intent_emb, slot_emb):
        super(Label_Attention, self).__init__()

        self.W_intent_emb = intent_emb.weight
        self.W_slot_emb = slot_emb.weight

    def forward(self, input_intent, input_slot, mask):
        intent_score = torch.matmul(input_intent, self.W_intent_emb.t())
        slot_score = torch.matmul(input_slot, self.W_slot_emb.t())
        intent_probs = nn.Softmax(dim=-1)(intent_score)
        slot_probs = nn.Softmax(dim=-1)(slot_score)
        intent_res = torch.matmul(intent_probs, self.W_intent_emb)
        slot_res = torch.matmul(slot_probs, self.W_slot_emb)

        return intent_res, slot_res


class I_S_SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, attention_dropout, max_seq_len):
        super(I_S_SelfAttention, self).__init__()

        self.num_attention_heads = 12
        self.attention_head_size = int(hidden_size / self.num_attention_heads)

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.out_size = out_size
        self.query = nn.Linear(input_size, self.all_head_size)
        self.query_slot = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.key_slot = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.out_size)
        self.value_slot = nn.Linear(input_size, self.out_size)
        self.dropout = nn.Dropout(attention_dropout)
        self.max_seq_len = max_seq_len

        self.RoE_QAndK = RoFormerSinusoidalPositionalEmb(self.max_seq_len, self.attention_head_size)

    def transpose_for_scores(self, x):
        last_dim = int(x.size()[-1] / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, last_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def roEmb(self, x, sinusoidal_pos):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        sin, cos = sinusoidal_pos
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return out

    def forward(self, intent, slot, mask):
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask = (1.0 - extended_attention_mask) * -10000.0

        mixed_query_layer = self.query(intent)
        mixed_key_layer = self.key(slot)
        mixed_value_layer = self.value(slot)

        mixed_query_layer_slot = self.query_slot(slot)
        mixed_key_layer_slot = self.key_slot(intent)
        mixed_value_layer_slot = self.value_slot(intent)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        query_layer_slot = self.transpose_for_scores(mixed_query_layer_slot)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        key_layer_slot = self.transpose_for_scores(mixed_key_layer_slot)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        value_layer_slot = self.transpose_for_scores(mixed_value_layer_slot)

        sinusoidal_pos = self.RoE_QAndK[None, None, :, :].chunk(2, dim=-1)

        query_layer = self.roEmb(query_layer, sinusoidal_pos)
        query_layer_slot = self.roEmb(query_layer_slot, sinusoidal_pos)
        key_layer = self.roEmb(key_layer, sinusoidal_pos)
        key_layer_slot = self.roEmb(key_layer_slot, sinusoidal_pos)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores_slot = torch.matmul(query_layer_slot, key_layer_slot.transpose(-1, -2))
        attention_scores_slot = attention_scores_slot / math.sqrt(self.attention_head_size)

        attention_scores_intent = attention_scores + attention_mask
        attention_scores_slot = attention_scores_slot + attention_mask

        attention_probs_slot = nn.Softmax(dim=-1)(attention_scores_slot)
        attention_probs_intent = nn.Softmax(dim=-1)(attention_scores_intent)

        attention_probs_slot = self.dropout(attention_probs_slot)
        attention_probs_intent = self.dropout(attention_probs_intent)

        context_layer_slot = torch.matmul(attention_probs_slot, value_layer_slot)
        context_layer_intent = torch.matmul(attention_probs_intent, value_layer)

        context_layer = context_layer_slot.permute(0, 2, 1, 3).contiguous()
        context_layer_intent = context_layer_intent.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.out_size,)
        new_context_layer_shape_intent = context_layer_intent.size()[:-2] + (self.out_size,)

        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer_intent = context_layer_intent.view(*new_context_layer_shape_intent)

        return context_layer, context_layer_intent


def Label2id(slot_label_list):
    xlen = len(slot_label_list)
    list_num = list(range(0, xlen))
    dict2 = {key: value for key, value in zip(slot_label_list, list_num)}
    return dict2


def RoFormerSinusoidalPositionalEmb(num_positions, embedding_dim):
    n_pos, dim = num_positions, embedding_dim
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
            for pos in range(n_pos)
        ]
    )
    out = torch.empty(num_positions, embedding_dim)
    sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
    out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    return out.to('cuda')
