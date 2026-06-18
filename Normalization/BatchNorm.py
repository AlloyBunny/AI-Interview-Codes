import torch
import torch.nn as nn

class BatchNorm(nn.Module):
    def __init__(self, C, eps=1e-5):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(1, C, 1, 1))
        self.bias  = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.eps = eps

    def forward(self, x):
        """
        x: [B, C, H, W], where:
            B: batch size
            C: number of channels
            H: height
            W: weight
        """

        mean = x.mean(dim=(0, 2, 3), keepdim=True)                # [B, C, H, W] -> [1, C, H, W]
        var  = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False) # [B, C, H, W] -> [1, C, H, W]

        x = (x - mean) * torch.rsqrt(var + self.eps)

        return x * self.weight + self.bias