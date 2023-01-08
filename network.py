import torch
import torch.nn as nn


class Siamese(nn.Module):
    def __init__(self, in_channels):
        super(Siamese, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.drop = nn.Dropout(0.25)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 50 * 50, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 50),
        )

    def forward(self, x):
        f1 = self.relu(self.conv1(x))
        f1 = self.pool(f1)
        f1 = self.drop(f1)
        f2 = self.relu(self.conv2(f1))
        f2 = self.pool(f2)
        f2 = self.drop(f2)
        v = self.fc(f2)
        return v
