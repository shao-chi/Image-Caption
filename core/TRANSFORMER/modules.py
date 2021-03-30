import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    
    def __init__(self, temperature, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, q, k, v, mask=None):

        attention = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attention = attention.masked_fill(mask, -np.inf)

        attention = self.softmax(attention)
        attention = self.dropout(attention)
        output = torch.matmul(attention, v)

        return output, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, input_size, q_k_dim, v_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.input_size = input_size
        self.q_k_dim = q_k_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.q_k_head_size = q_k_dim // num_heads
        self.v_head_size = v_dim // num_heads

        self.q_linear = nn.Linear(input_size, q_k_dim, bias=False)
        self.k_linear = nn.Linear(input_size, q_k_dim, bias=False)
        self.v_linear = nn.Linear(input_size, v_dim, bias=False)
        nn.init.normal_(tensor=self.q_linear.weight,
                        mean=0,
                        std=np.sqrt(2.0 / (input_size + q_k_dim)))
        nn.init.normal_(tensor=self.k_linear.weight,
                        mean=0,
                        std=np.sqrt(2.0 / (input_size + q_k_dim)))
        nn.init.normal_(tensor=self.v_linear.weight,
                        mean=0,
                        std=np.sqrt(2.0 / (input_size + v_dim)))

        self.attention = \
                ScaledDotProductAttention(temperature=self.q_k_head_size ** 0.5)
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)

        self.joint_linear = nn.Linear(in_features=num_heads * self.v_head_size,
                                      out_features=input_size,
                                      bias=False)
        nn.init.xavier_normal_(self.joint_linear.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):
        residual = q

        sz_b, len_q, _ = q.size()

        q = self.q_linear(q) \
                .view(q.size(0), q.size(1), self.num_heads, self.q_k_head_size)
        k = self.k_linear(k) \
                .view(k.size(0), k.size(1), self.num_heads, self.q_k_head_size)
        v = self.v_linear(v) \
                .view(v.size(0), v.size(1), self.num_heads, self.v_head_size)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        output, attention = self.attention(q, k, v, mask=mask)

        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        output = self.joint_linear(output)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output, attention


class FeedForward(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(FeedForward, self).__init__()

        self.position_wise_1 = nn.Linear(input_size, hidden_size)
        self.position_wise_2 = nn.Linear(hidden_size, input_size)
        nn.init.xavier_normal_(self.position_wise_1.weight)
        nn.init.xavier_normal_(self.position_wise_2.weight)

        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()


    def forward(self, x):
        residual = x
        # output = x.transpose(1, 2)

        output = self.position_wise_1(x)
        output = self.relu(output)

        output = self.position_wise_2(output)
        # output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class EncoderBlock(nn.Module):

    def __init__(self,
                input_size,
                hidden_size,
                num_heads,
                q_k_dim,
                v_dim,
                dropout=0.1):
        super(EncoderBlock, self).__init__()

        self.multihead_attention = MultiHeadAttention(input_size=input_size,
                                                      q_k_dim=q_k_dim,
                                                      v_dim=v_dim,
                                                      num_heads=num_heads,
                                                      dropout=dropout)
        self.feed_forward = FeedForward(input_size=input_size,
                                        hidden_size=hidden_size,
                                        dropout=dropout)


    def forward(self, encode_input, non_pad_mask=None, attention_mask=None):
        output, attention = self.multihead_attention(q=encode_input,
                                                     k=encode_input,
                                                     v=encode_input,
                                                     mask=attention_mask)

        output = self.feed_forward(output)
        
        if non_pad_mask is not None:
            output *= non_pad_mask

        return output, attention


class DecoderBlock(nn.Module):

    def __init__(self,
                input_size,
                hidden_size,
                num_heads,
                q_k_dim,
                v_dim,
                dropout=0.1):
        super(DecoderBlock, self).__init__()

        self.self_attention = MultiHeadAttention(input_size=input_size,
                                                 q_k_dim=q_k_dim,
                                                 v_dim=v_dim,
                                                 num_heads=num_heads,
                                                 dropout=dropout)
        self.encode_attention = MultiHeadAttention(input_size=input_size,
                                                   q_k_dim=q_k_dim,
                                                   v_dim=v_dim,
                                                   num_heads=num_heads,
                                                   dropout=dropout)
        self.feed_forward = FeedForward(input_size=input_size,
                                        hidden_size=hidden_size,
                                        dropout=dropout)

    def forward(self, decode_input, encode_output,
                non_pad_mask=None,
                self_attention_mask=None,
                context_attention_mask=None):
        decode_output, decode_attention = \
                self.self_attention(q=decode_input,
                                    k=decode_input,
                                    v=decode_input,
                                    mask=self_attention_mask)

        decode_output, context_attention = \
                self.encode_attention(q=decode_output,
                                      k=encode_output,
                                      v=encode_output,
                                      mask=context_attention_mask)

        decode_output = self.feed_forward(decode_output)

        if non_pad_mask is not None:
            decode_output *= non_pad_mask

        return decode_output, decode_attention, context_attention