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
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        x: [batch_size, seq_len, embed_dim]
        attn_mask: [batch_size, seq_len]
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

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask[:, None, None, :] == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        output = attn @ v  # [batch_size, num_heads, seq_len, head_dim]
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        return output
