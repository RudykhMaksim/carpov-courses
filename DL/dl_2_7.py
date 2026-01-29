import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch.nn as nn

def train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn):
    model.train()  # Переводим модель в режим обучения
    total_loss = 0.0
    num_batches = len(data_loader)
    
    for batch in data_loader:
        # Предполагаем, что batch содержит входные данные и цели
        inputs, targets = batch
        
        optimizer.zero_grad()  # Обнуляем градиенты
        
        outputs = model(inputs)  # Проход вперед
        loss = loss_fn(outputs, targets)  # Вычисляем ошибку
        
        loss.backward()  # Проход назад
        print(f"{loss.item():.5f}")  # Печатаем ошибку с точностью до 5 символов после запятой

        optimizer.step()  # Шаг оптимизации
        
        total_loss += loss.item()  # Суммируем ошибку
    
    average_loss = total_loss / num_batches  # Вычисляем среднюю ошибку
    return average_loss
