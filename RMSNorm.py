import torch
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.eplsilon = 1e-8
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        
    def forward(self, x):
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = x / torch.sqrt(var + self.eplsilon)
        return self.gamma * norm_x 