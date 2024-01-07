import torch.nn as nn
import AttentionHead


''' Multi-head block for multi-head self attention '''
class MultiHeadBlock(nn.Module):
  def __init__(self, n_tokens, n_features, n_heads, dropout=0.0) -> None:
    super().__init__()

    # ensure n_features is divisible by n_heads so head_size is correct
    # ensures that the output of forward will be of correct dimensions
    try:
      if n_features % n_heads != 0:
        raise Exception("n_features must be divisible by n_heads")
    except Exception as e:
      print(e)

    head_size = n_features // n_heads
    self.heads = nn.ModuleList([AttentionHead(n_tokens, n_features, head_size, dropout) for i in range(n_heads)])
    self.proj = nn.Linear(n_heads * head_size, n_features) #!!!!!Projection
    self.dropout = nn.Dropout(dropout)

  def forward(self, input):
    out = torch.cat([head(input) for head in self.heads], dim=-1) # B, T, F
    out = self.proj(out) #!!!!!Projection
    out = self.dropout(out)
    return out