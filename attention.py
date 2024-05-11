import torch
import torch.nn as nn
from linear import CustomLinear
import torch.nn.functional as F

class Head(nn.Module):
  """Implements a single head of attention."""

  def __init__(self, head_size, batch_size, tokens_per_batch, channel_size, head_type='decoder', dropout=0.2):
    super().__init__()
    self.head_size = head_size

    self.batch_size = batch_size
    self.tokens_per_batch = tokens_per_batch
    self.channel_size = channel_size
    self.head_type = head_type

    self.key = CustomLinear(channel_size, head_size) # multiple a token tensoer to get what I have
    self.query = CustomLinear(channel_size, head_size) # multiple a token tensoer to get what I want
    self.value = CustomLinear(channel_size, head_size) # multiple a token tensoer to get what I will give you

    self.dropout = nn.Dropout(dropout)


  def forward(self, x):

    # x is batch_size, tocken_per_batch, channel_size

    k = self.key(x) # batch_size, tocken_per_batch, head_size
    q = self.query(x) # batch_size, tocken_per_batch, head_size

    # how similar are keys and queries
    w = k @ q.transpose(-2, -1) # batch_size, tocken_per_batch, head_size @ batch_size, head_size, tocken_per_batch = batch_size, tocken_per_batch, tocken_per_batch

    # do we want to mask?
    if self.head_type == 'decoder':
      tril = torch.tril(torch.ones(self.tokens_per_batch, self.tokens_per_batch))
      w.masked_fill_(tril == 0, float('-inf'))

    # normalize the weights as probablity
    w = F.softmax(w, dim=-1)
    w = self.dropout(w)

    v = self.value(x) # batch_size, tocken_per_batch, head_size
    out = w @ v # batch_size, tocken_per_batch, head_size
    return out
  


class MultiHeadAttention(nn.Module):
  """Implements a multiple heads of attention working in parrallel"""

  def __init__(self, num_heads, head_size, batch_size, tokens_per_batch, channel_size, head_type='decoder', dropout=0.2):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size, batch_size, tokens_per_batch, channel_size, head_type=head_type) for _ in range(num_heads)])
    self.projection = CustomLinear(channel_size, channel_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([head(x) for head in self.heads], dim=-1)
    out = self.projection(out)
    out = self.dropout(out)
    return out
