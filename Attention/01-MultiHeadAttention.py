import torch
import torch.nn as nn
import math


class MultiHeadMaskedSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=4096, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # QKV 投影层：将输入映射到统一的 d_model 空间
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # 输出投影：用于融合多头信息
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # attention dropout（防止注意力过拟合）
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        x: [B, L, d_model]
        attention_mask: [B, L]
            - 1 表示有效 token
            - 0 表示 padding token
        """
        B, L, D = x.shape

        # 1) QKV 线性映射
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2) 拆分 multi-head
        # [B, L, d_model] -> [B, L, H, D] -> [B, H, L, D]
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) scaled dot-product attention
        # 为什么要除 sqrt(d_k)：防止 dot product 随维度变大导致 softmax 饱和
        # scores: [B, H, L, L]
        scores = (Q @ K.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # 4) attention mask
        if attn_mask is not None:
            scores = scores.masked_fill(
                attn_mask[:, None, None, :] == 0,
                float("-inf")
            )

        # 5) softmax 得到 attention 权重
        attn = torch.softmax(scores, dim=-1)

        # 6) attention dropout（Transformer 标配正则化）
        attn = self.attn_drop(attn)

        # 7) 加权求和
        output = attn @ V  # [B, H, L, D]

        # 8) 多头拼接
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)

        # 9) 输出投影（融合 multi-head 信息）
        output = self.W_o(output)

        return output