from src.entity.config_entity import ModelConfig
from torch import nn
import torch


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = ModelConfig()
        self.base_model = self.get_model()
        self.conv1 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.flatten = nn.Flatten()
        self.final = nn.Linear(4 * 8 * 8, self.config.LABEL)

    def get_model(self):
        torch.hub.set_dir(self.config.STORE_PATH)
        model = torch.hub.load(
            self.config.REPOSITORY,
            self.config.BASEMODEL,
            pretrained=self.config.PRETRAINED
        )
        return nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        x = self.base_model(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.final(x)
        return x


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = NeuralNet()
    net.to(device)
