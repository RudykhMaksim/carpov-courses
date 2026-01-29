import torch
import torch.nn as nn

def create_model():
    model = nn.Sequential(
        nn.Linear(100, 10),  # Входной слой: 100 входных нейронов, 10 выходных нейронов
        nn.ReLU(),           # Нелинейная активация ReLU
        nn.Linear(10, 1)     # Выходной слой: 10 входных нейронов, 1 выходной нейрон
    )
    return model
