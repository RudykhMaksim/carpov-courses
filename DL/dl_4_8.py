import torch
import torch.nn as nn


def create_simple_conv_cifar():
    class SimpleConvNet(nn.Module):
        def __init__(self):
            super(SimpleConvNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(2, 2)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(64 * 8 * 8, 512)
            self.fc2 = nn.Linear(512, 10)

        def forward(self, x):
            x = self.pool(self.relu(self.bn1(self.conv1(x))))
            x = self.pool(self.relu(self.bn2(self.conv2(x))))
            x = x.view(-1, 64 * 8 * 8)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    return SimpleConvNet()
