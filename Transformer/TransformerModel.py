import torch.nn as nn
import Transformer
import FloatEmbedding
import PositionalEncoder
import math

''' Decoder transformer nueral network model, has lower-triangular mask in self attention '''
class TransformerNN(nn.Module):
  def __init__(self, vocab_size, batch_size, n_tokens, n_features, n_heads, n_layers, dropout=0.0, learning_rate=3e-5) -> None:
    super().__init__()

    #data
    self.train_data = None
    self.val_data = None
    self.batch_size = batch_size
    self.n_tokens = n_tokens
    self.eval_interval = None
    self.eval_iters = None

    # with torch.no_grad():
    #   self.loss_estimator = LossEstimator()

    #dropout
    self.dropout = nn.Dropout(dropout)

    # layers
    self.embedding = FloatEmbedding.FloatEmbedding(n_features)
    self.positions_enc = PositionalEncoder(n_tokens, n_features, dropout)
    self.transformers = nn.Sequential(*[Transformer(n_tokens, n_features, n_heads, dropout) for i in range(n_layers)])
    self.ln_f = nn.LayerNorm(n_features)
    self.lin = nn.Linear(n_features, 1)

    # initialize weights
    self.apply(self._init_weights)
    # apply special scaled init to the residual projections, per GPT-2 paper
    for pn, p in self.named_parameters():
        if pn.endswith('proj.weight'):
            nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layers))

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, input):
    out = self.embedding(input)
    out = self.positions_enc(out)
    out = self.transformers(out)
    out = self.ln_f(out)
    out = self.lin(out)

    return out