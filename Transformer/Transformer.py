import torch.nn as nn
import MultiHeadBlock
import FeedForwardBlock

''' Decoder transformer, has lower-triangular mask in self attention '''
class Transformer(nn.Module):
  def __init__(self, n_tokens, n_features, n_heads, dropout=0.0) -> None:
    super().__init__()
    self.ln1 = nn.LayerNorm(n_features)
    self.multi_head_block = MultiHeadBlock(n_tokens, n_features, n_heads, dropout)
    self.ln2 = nn.LayerNorm(n_features)
    self.feed_forward_block = FeedForwardBlock(n_features, dropout)

  def forward(self, input):
    out = self.multi_head_block(self.ln1(input)) + input
    out = self.feed_forward_block((self.ln2(out))) + out
    return out