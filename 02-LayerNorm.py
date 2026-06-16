import torch
import torch.nn as nn

"""
02-LayerNorm
"""
class LayerNorm(nn.Module):
    def __init__(self, C, H, W): 
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(C, H, W))
        self.beta = nn.Parameter(torch.zeros(C, H, W))
    
    def forward(self, x):

        mean = x.mean((1,2,3), keepdim=True)
        var  = x.var((1,2,3), keepdim=True, unbiased=False)

        x = (x - mean) / torch.sqrt(var + 1e-5)

        return self.gamma[None,:,:,:] * x + self.beta[None,:,:,:]