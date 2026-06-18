import torch
import torch.nn as nn
import math

"""
SelfAttention
实现一个SelfAttention层，输入是[B, L, d_model]，输出是[B, L, d_model]，其中B是Batch Size，L是Sequence Length，d_model是Embedding Dimension。
其中，SelfAttention的公式为：
Q = x @ W_q
K = x @ W_k
V = x @ W_v

scores = Q @ K.T / sqrt(d_k)
attn = softmax(scores)
output = attn @ V
注：
1. 这里实现的是Encoder self-attention，没有causal mask
2. d_k和d_v一般是相等的；在MHA里，一般n*d_k=d_model（n是头数）
"""

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        # 这里需要特别注意，声明一个线性层的语法是nn.Linear(in_dim, out_dim)
        # 而它对应的权重矩阵的维度是[out_dim, in_dim]，偏置项的维度是[out_dim]
        self.W_q = nn.Linear(d_model, d_k, bias=False) # bias=False的意思是，y=x@W.T+b中，不加偏置项b，只有y=x@W.T，更符合论文中的公式
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)
    
    def forward(self, x):
        """
        x: [B, L, d_model]
        """
        Q = self.W_q(x)     # Q: [B, L, d_k]
        K = self.W_k(x)     # K: [B, L, d_k]
        V = self.W_v(x)     # V: [B, L, d_v]

        scores = Q @ K.transpose(-1, -2) / math.sqrt(K.size(-1))    # scores: [B, L, L]
        attn = torch.softmax(scores, dim=-1)                        # attn: [B, L, L]
        output = attn @ V                                           # output: [B, L, d_v]
        return output