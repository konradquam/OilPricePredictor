import torch.nn as nn
from torch.nn import functional as F

''' Head for self attention '''
class AttentionHead(nn.Module):
  def __init__(self, n_tokens, n_features, head_size, dropout=0.0) -> None:
    super().__init__()
    self.q = nn.Linear(n_features, head_size)
    self.k = nn.Linear(n_features, head_size)
    self.v = nn.Linear(n_features, head_size)
    self.register_buffer('tril', torch.tril(torch.ones(n_tokens, n_tokens)))

    self.dropout = nn.Dropout(dropout)
  def forward(self, input):
    T = input.shape[1]
    query = self.q(input) # B, T, hs
    key = self.k(input) # B, T, hs
    value = self.v(input) # B, T, hs
    weights = key @ query.transpose(1, 2) * key.shape[-1]**-0.5 # B, T, T
    weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    weights = F.softmax(weights, dim = -1) #softmax across rows for each batch, since each row corresponds to token
    weights = self.dropout(weights)
    out = weights @ value # B, T, hs
    return out