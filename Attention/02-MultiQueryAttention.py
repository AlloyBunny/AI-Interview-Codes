import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

# 注：MQA只是GQA中num_kv_heads=1的特殊情况，其实只学GQA就够
class MultiQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=False)

        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, embed_dim = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # q: [batch_size, seq_len, embed_dim]
        # -> [batch_size, seq_len, num_heads, head_dim]
        # -> [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # k/v: [batch_size, seq_len, head_dim]
        #   -> [batch_size, 1, seq_len, head_dim]
        k = k.unsqeeze(1)
        v = v.unsqeeze(1)

        # scores: [batch_size, num_heads, seq_len, seq_len]
        scores = q @ k.transpose(-1,-2) / math.sqrt(self.head_dim)

        # causal_mask
        scores = scores + torch.triu(
            torch.full((seq_len, seq_len),float('-inf'), device=scores.device), # 创造一个全是-inf的[seq_len, seq_len]矩阵
            diagonal=1
        )[None, None, :, :]

        if padding_mask is not None:
            scores += (1.0 - padding_mask[:, None, None, :]) * -1e9

        # attn: [batch_size, num_heads, seq_len, seq_len]
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # output: [batch_size, num_heads, seq_len, haed_dim]
        output = attn @ v
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        output = self.o_proj(output)

        return output