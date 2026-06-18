import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    out = torch.empty_like(x)

    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos

    return out


def slice_rope(position_embedding: Tuple[torch.Tensor, torch.Tensor], start: int, length: int):
    cos, sin = position_embedding

    # 支持两种常见形状：
    # 1) [max_len, head_dim // 2]
    # 2) [1, max_len, head_dim // 2]
    if cos.dim() >= 3 and cos.size(0) == 1:
        return cos[:, start:start + length], sin[:, start:start + length]
    return cos[start:start + length], sin[start:start + length]


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        assert num_heads % num_kv_heads == 0, "num_heads 必须能被 num_kv_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_groups = num_heads // num_kv_heads
        assert self.head_dim % 2 == 0, "RoPE 要求 head_dim 必须是偶数"

        # Q 仍然有 num_heads 个头
        self.W_q = nn.Linear(d_model, d_model, bias=False)

        # K/V 只有 num_kv_heads 个头，所以输出维度更小
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)

        # 输出投影：用于融合多头信息
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # attention dropout（防止注意力过拟合）
        self.attn_drop = nn.Dropout(dropout)

    def forward(
            self, x: torch.Tensor, position_embedding: Tuple[torch.Tensor, torch.Tensor],
            past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False,
            attn_mask: Optional[torch.Tensor] = None
        ):
        """
        x: [B, L, d_model]
        position_embedding: 一般为完整 RoPE 表，需要在这里根据 past_len 切片
        past_kv: Optional[Tuple[K, V]]
            - K/V: [B, H_kv, past_len, head_dim]
        attn_mask: [B, total_len]
            - 1 表示有效 token
            - 0 表示 padding token
        """
        B, L, D = x.shape

        # 1) QKV 线性映射
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2) 拆分 query heads
        # Q: [B, L, D] -> [B, L, H, D] -> [B, H, L, D]
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) 拆分 key/value heads
        # K/V: [B, L, H_kv * D] -> [B, L, H_kv, D] -> [B, H_kv, L, D]
        K = K.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 4) 对Q/K使用RoPE位置编码
        # 有 KV cache 时，当前 token 的位置不是从 0 开始，而是从 past_len 开始
        past_len = 0 if past_kv is None else past_kv[0].size(2)
        cos, sin = slice_rope(position_embedding, past_len, L)
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        if past_kv is not None:
            K = torch.cat([past_kv[0], K], dim=2)  # 在序列长度维度上拼接
            V = torch.cat([past_kv[1], V], dim=2)
        present_kv = (K, V) if use_cache else None
        total_len = K.size(2)

        # 5) 将每个 K/V head 复制给同一组里的多个 Q head
        # [B, H_kv, total_len, D] -> [B, H, total_len, D]
        # 相当于 K = K[:, :, None, :, :].expand(B, H_kv, G, total_len, D).reshape(B, H, total_len, D)
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)

        # 6) scaled dot-product attention
        # 为什么要除 sqrt(d_k)：防止 dot product 随维度变大导致 softmax 饱和
        # scores: [B, H, L, total_len]
        scores = (Q @ K.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # 7) attention mask
        if attn_mask is not None:
            scores = scores.masked_fill(
                attn_mask[:, None, None, :] == 0,
                float("-inf")
            )

        # 8) softmax 得到 attention 权重
        attn = torch.softmax(scores, dim=-1)

        # 9) attention dropout（Transformer 标配正则化）
        attn = self.attn_drop(attn)

        # 10) 加权求和
        output = attn @ V  # [B, H, L, D]

        # 11) 多头拼接
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)

        # 12) 输出投影（融合 multi-head 信息）
        output = self.W_o(output)

        return output, present_kv
