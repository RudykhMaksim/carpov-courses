import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block with skip-connection
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        return F.relu(self.block(x) + x)

# Improved CIFAR10 model with skip connections
class AdvancedSkipConnectionCNN(nn.Module):
    def __init__(self):
        super(AdvancedSkipConnectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(64, 3)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer4 = self._make_layer(128, 3)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def _make_layer(self, channels, num_blocks):
        layers = [ResidualBlock(channels) for _ in range(num_blocks)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def create_advanced_skip_connection_conv_cifar():
    return AdvancedSkipConnectionCNN()


