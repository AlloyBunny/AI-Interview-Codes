import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()

        assert dim % 2 == 0, "RoPE dimension must be even"
        self.dim = dim

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)) # 笔记中的θ，旋转频率
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        B, T, D = x.shape
        device = x.device

        pos = torch.arange(T, device=device).float() # 笔记中的p，表示是sequence中第p个token

        # 外积（替代 einsum）
        freqs = pos[:, None] * self.inv_freq[None, :] # 可以理解为for i, for p, calculate 旋转角 = p * θ_i

        cos = freqs.cos()[None, :, :] # x.cos()就是[cos(x_i) for x_i in x], x.sin()同理
        sin = freqs.sin()[None, :, :] # 这里的[None, :, :]就是unsqeeze(0)的意思，[T, D]->[1, T, D]，方便后续boardcast

        x1 = x[..., 0::2] # [x_0, x_2, x_4, ...]
        x2 = x[..., 1::2] # [x_1, x_3, x_5, ...]

        out = torch.empty_like(x) # 形状和x一样的zeros

        out[..., 0::2] = x1 * cos - x2 * sin # 取out的对应位置写入，这里公式是对应逆时针旋转freqs角度
        out[..., 1::2] = x1 * sin + x2 * cos

        return out