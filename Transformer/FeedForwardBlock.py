import torch.nn as nn

''' Feed-forward block '''
class FeedForwardBlock(nn.Module):
  scale_factor = 4
  def __init__(self, n_features, dropout=0.0) -> None:
    super().__init__()
    self.proj = nn.Linear(n_features * self.scale_factor, n_features)
    self.layers = nn.Sequential(
        nn.Linear(n_features, n_features * self.scale_factor),
        nn.GELU(),
        self.proj,
        nn.Dropout(dropout)
    )

  def forward(self, input):
    out = self.layers(input)
    return out
