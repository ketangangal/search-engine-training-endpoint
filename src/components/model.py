from torch import nn
from from_root import from_root
import os
import torch

# Require final layer to be dynamic 
# Require one path to store the model
# While embedding we need liner layers to be 256 as Dimension Reduction.

class NeuralNet(nn.Module):
  def __init__(self, labels):
    super().__init__()
    torch.hub.set_dir(os.path.join(from_root(), "model","benchmark"))
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    self.base_model = nn.Sequential(*list(model.children())[:-2])
    self.flatten = nn.Flatten()
    self.final = nn.Linear(512*8*8,labels)

  def forward(self,x):
    x = self.base_model(x)
    x = self.flatten(x)
    x = self.final(x)
    return x


if __name__ == '__main__':
  device = "cuda" if torch.cuda.is_available() else "cpu"
  net = NeuralNet()
  net.to(device)