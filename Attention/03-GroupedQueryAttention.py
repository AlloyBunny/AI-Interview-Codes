import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    x: [batch_size, num_heads, seq_len, head_dim]
    cos/sin: [seq_len, head_dim // 2]
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos

    return out

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        assert num_heads % num_kv_heads == 0, "num_heads 必须能被 num_kv_heads 整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        assert self.head_dim % 2 == 0, "RoPE 要求 head_dim 必须是偶数"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        position_embedding: Tuple[torch.Tensor, torch.Tensor],
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        x: [batch_size, seq_len, embed_dim]
        position_embedding: Tuple[cos, sin]
            - cos/sin: [max_len, head_dim // 2]
        past_kv: Optional[Tuple[k, v]]
            - k/v: [batch_size, num_kv_heads, past_len, head_dim]
        use_cache:
            - 是否返回新的 KV cache
        attn_mask: [batch_size, total_len]
            - 1 表示有效 token
            - 0 表示 padding token
        """
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, "x.size(-1) 必须等于 embed_dim"

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # q: [batch_size, seq_len, embed_dim]
        # -> [batch_size, seq_len, num_heads, head_dim]
        # -> [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # k/v: [batch_size, seq_len, num_kv_heads * head_dim]
        # -> [batch_size, seq_len, num_kv_heads, head_dim]
        # -> [batch_size, num_kv_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        past_len = 0 if past_kv is None else past_kv[0].size(2)
        cos, sin = position_embedding
        cos = cos[past_len:past_len + seq_len]
        sin = sin[past_len:past_len + seq_len]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_kv is not None:
            # k/v: [batch_size, num_kv_heads, seq_len, head_dim]
            #   -> [batch_size, num_kv_heads, total_len, head_dim] (其中total_len = past_len + seq_len)
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        past_kv = (k, v) if use_cache else None

        # 每个 k/v head 复制给同一组里的多个 q head。
        # [batch_size, num_kv_heads, total_len, head_dim]
        # -> [batch_size, num_heads, total_len, head_dim]
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        # scores: [batch_size, num_heads, seq_len, total_len]
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask[:, None, None, :] == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # output: [batch_size, num_heads, seq_len, head_dim]
        output = attn @ v
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        return output, past_kv