import torch
import torch.nn as nn
import math


class MultiHeadMaskedSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 这里和单头版本一样，仍然用 Linear
        # 只是现在先把输入映射到总维度 d_model，再拆成多个 head
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # 多头拼接后，再过一个输出投影层
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        """
        x: [B, L, d_model]
        """
        B, L, _ = x.shape

        # 1) 线性映射
        # Q/K/V: [B, L, d_model] @ [d_model, d_model] = [B, L, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2) 拆成多头
        # Q: [B, L, d_model] -> [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) 计算 attention scores
        # 和self-attention几乎一样，把d_k换成head_dim
        # scores: [B, num_heads, L, L]
        scores = Q @ K.transpose(-1, -2) / math.sqrt(self.head_dim)

        # 4) 默认使用 causal mask（下三角），也可以根据具体情况换成别的mask
        mask = torch.tril(torch.ones(L, L, device=x.device))  # mask: [L, L]
        mask = mask.unsqueeze(0).unsqueeze(0)                 # mask: [1, 1, L, L]，方便后面广播
        scores = scores.masked_fill(mask == 0, float("-inf")) # mask==0 的位置不能看，填成负无穷

        # 5) softmax 得到注意力权重
        # attn: [B, num_heads, L, L]
        attn = torch.softmax(scores, dim=-1)

        # 6) 加权求和
        # output: [B, num_heads, L, head_dim]
        output = attn @ V

        # 7) 多头拼接回去
        # output: [B, num_heads, L, head_dim] -> [B, L, num_heads, head_dim] -> [B, L, d_model]
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)

        # 8) 输出投影
        # output: [B, L, d_model]
        output = self.W_o(output)

        return output