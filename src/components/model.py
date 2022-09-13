from torch import nn
import torch

# Require final layer to be dynamic 
# Require one path to store the model
# While embedding we need liner layers to be 256 as Dimension Reduction.

class NeuralNet(nn.Module):
  def __init__(self, layers):
    super().__init__()
    self.base_model = layers
    self.flatten = nn.Flatten()
    self.linear = nn.Linear(512*7*7,101)

  def forward(self,x):
    x = self.base_model(x)
    x = self.flatten(x)
    x = self.linear(x)
    return x


if __name__ == '__main__':
  device = "cuda" if torch.cuda.is_available() else "cpu"
  net = NeuralNet()
  net.to(device)