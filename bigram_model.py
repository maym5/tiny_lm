from linear import CustomLinear
import torch
import torch.nn as nn
from transformer import TransformerBlock
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):

  def __init__(self, vocab, attention_block_size, n_embed=32, depth=8):
    super().__init__()
    self.vocab = vocab
    vocab_size = len(vocab)
    self.embeddings_table_ = nn.Embedding(vocab_size, n_embed)
    self.positional_embedding_table_ = nn.Embedding(attention_block_size, n_embed)
    self.layers = nn.Sequential(*[TransformerBlock(4, n_embed, vocab_size, 4, 8) for _ in range(depth)])
    self.lm = CustomLinear(n_embed, vocab_size)


  def forward(self, x, y=None):
    B, T = x.shape
    tok_emb = self.embeddings_table_(x)
    pos_emb = self.positional_embedding_table_(torch.arange(T))
    x = tok_emb + pos_emb
    logits = self.layers(x.to(torch.float))
    logits = self.lm(logits)

    if y is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T,C)
      y = y.view(B*T)
      loss = F.cross_entropy(logits, y)

    return logits, loss


  def generate(self, idx, max_new_tokens):
    decode = lambda x: ''.join([self.vocab[xi] for xi in x])
    for _ in range(max_new_tokens):
      logits, loss = self(idx)

      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)

      idx = torch.cat((idx, idx_next), dim=1)
    return decode(idx.to(torch.long).flatten())
