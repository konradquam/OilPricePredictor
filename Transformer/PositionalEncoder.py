import torch

'''
Does the positional encoding
Code based on Medium article: TODO find the medium article for credit
'''
class PositionalEncoder(nn.Module):

  def __init__(self, n_tokens, n_features, dropout):
    super().__init__()

    self.dropout = nn.Dropout(dropout)

    #positional encoding
    pos_encoding = torch.zeros(n_tokens, n_features)
    positions_list = torch.arange(0, n_tokens, dtype=torch.float).view(-1, 1)
    division_term = torch.exp(torch.arange(0, n_features, 2).float() * (-math.log(10000.0)) / n_features)

    #input even
    pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

    #input odd
    pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

    #unsqueez
    pos_encoding = pos_encoding.unsqueeze(0).transpose(0,1)

    #save buffer
    self.register_buffer("pos_encoding",pos_encoding)

  def forward(self, token_embedding):
     # Residual connection + pos encoding
      return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])