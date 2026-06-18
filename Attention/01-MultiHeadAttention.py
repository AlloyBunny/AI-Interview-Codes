import math

import torch
import torch.nn as nn


class MultiHeadMaskedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        """
        x: [batch_size, seq_len, embed_dim]
        padding_mask: [batch_size, seq_len]
            - 1 表示有效 token
            - 0 表示 padding token
        """
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, "x.size(-1) 必须等于 embed_dim"

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [batch_size, seq_len, embed_dim]
        # -> [batch_size, seq_len, num_heads, head_dim]
        # -> [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)  # [batch_size, num_heads, seq_len, seq_len]

        # causal_mask
        scores = scores + torch.triu(
            torch.full((seq_len, seq_len),float('-inf'), device=scores.device), # 创造一个全是-inf的[seq_len, seq_len]矩阵
            diagonal=1
        )[None, None, :, :]

        if padding_mask is not None:
            # 假设padding_mask的定义是：为0的位置是要mask掉的，1是保留的
            # 下面这句等效于scores = scores.masked_fill(padding_mask[:, None, None, :] == 0, float("-inf"))
            scores += (1.0 - padding_mask[:, None, None, :]) * -1e9

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        output = attn @ v  # [batch_size, num_heads, seq_len, head_dim]
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        output = self.o_proj(output)

        return output
