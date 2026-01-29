import torch
from torch import nn

def create_conv_model():
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),     # [B, 1, 28, 28] → [B, 32, 28, 28]
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),    # → [B, 64, 28, 28]
        nn.ReLU(),
        nn.MaxPool2d(2),                                # → [B, 64, 14, 14]
        nn.Conv2d(64, 128, kernel_size=3, padding=1),   # → [B, 128, 14, 14]
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),  # → [B, 128, 14, 14]
        nn.ReLU(),
        nn.MaxPool2d(2),                                # → [B, 128, 7, 7]
        nn.Flatten(),                                   # → [B, 128*7*7]
        nn.Linear(128 * 7 * 7, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


