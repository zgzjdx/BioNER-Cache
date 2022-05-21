import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.selfattention = nn.MultiheadAttention(embed_dim, num_heads=1, dropout=0.0, bias=True, add_bias_kv=False,
                                                   add_zero_attn=False, kdim=None, vdim=None)

    def forward(self, x):
        L, N, E = x.shape
        W1 = Parameter(torch.empty((L, N, N)))
        W2 = Parameter(torch.empty((L, N, N)))
        W3 = Parameter(torch.empty((L, N, N)))

        std = 1. / math.sqrt(self.embed_dim)

        nn.init.uniform_(W1, -std, std)
        nn.init.uniform_(W2, -std, std)
        nn.init.uniform_(W3, -std, std)

        query = W1.matmul(x)  # (L, N, E)
        key = W2.matmul(x)  # (S,N,E)
        value = W3.matmul(x)  # (S,N,E)
        attn_output, _ = self.selfattention(query, key, value)  # (L,N,E)
        return attn_output