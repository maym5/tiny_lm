import torch
import torch.nn as nn
import math

class CustomLinear(nn.Module):

  def __init__(self, in_features, out_features,  include_biases=False):

    super().__init__()
    self.in_features = in_features
    self.out_features = out_features

    weights = torch.empty(in_features, out_features)
    self.weights = nn.Parameter(weights)

    biases = torch.empty(out_features)
    self.biases = nn.Parameter(weights)
    self.include_biases = include_biases

    nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
    if include_biases:
      fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
      bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
      nn.init.uniform_(biases, -bound, bound)
    else:
      self.register_parameter('bias', None)


  def forward(self, x):
    """Implements the forward pass"""
    res = x @ self.weights
    if self.include_biases:
      return res + self.biases
    return res
