import torch
import torch.nn as nn
from attention import MultiHeadAttention
from layer_norm import LayerNorm1d
from linear import CustomLinear

class TransformerMLP(nn.Module):
  """The Mulit-layered Perceptron we want to include in our transformer block"""

  def __init__(self, embed_dim, vocab_size, dropout=0.2):
    super().__init__()

    self.layers = nn.Sequential(*[
        CustomLinear(embed_dim, 4 * embed_dim),
        nn.ReLU(),
        CustomLinear(4 * embed_dim, 4 * embed_dim),
        nn.ReLU(),
        CustomLinear(4 * embed_dim, embed_dim),
        nn.Dropout(dropout)])

  def forward(self, x):
    return self.layers(x)


class TransformerBlock(nn.Module):

  def __init__(self, num_heads, embed_dim, vocab_size, batch_size, tokens_per_batch, head_type='decoder'):

    super().__init__()
    self.attention = MultiHeadAttention(num_heads, embed_dim//num_heads, batch_size, tokens_per_batch, embed_dim, head_type)
    self.mlp = TransformerMLP(embed_dim, vocab_size)
    self.ln1 = LayerNorm1d(embed_dim)
    self.ln2 = LayerNorm1d(embed_dim)


  def forward(self, x):
    x = x + self.attention(self.ln1(x))
    return x + self.mlp(self.ln2(x))
