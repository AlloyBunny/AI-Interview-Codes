import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) # 注：llama/GPT系有些模型是weight only的，没有bias
        self.eps = eps
    
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True) # mean square of x
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight + self.bias