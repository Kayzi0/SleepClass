import torch
from torch import nn
from torchsummary import summary

class ConvNet (nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=11, out_channels=32, kernel_size=5, padding=2, stride = 2, bias=False),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            #nn.MaxPool1d(2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride = 2, bias=False),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            #nn.MaxPool1d(2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2, stride = 2, bias=False),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            #nn.MaxPool1d(2)
        )

        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=5)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        #x = self.convclass(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    summary(ConvNet(), (5, 300))