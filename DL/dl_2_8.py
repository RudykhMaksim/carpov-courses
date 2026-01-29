import torch
from torch.utils.data import DataLoader
import torch.nn as nn

def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn):
    model.eval()  # Переводим модель в режим инференса
    total_loss = 0.0
    num_batches = len(data_loader)
    
    with torch.no_grad():  # Не вычисляем градиенты для оценки
        for batch in data_loader:
            # Предполагаем, что batch содержит входные данные и цели
            inputs, targets = batch
            
            outputs = model(inputs)  # Проход вперед
            loss = loss_fn(outputs, targets)  # Вычисляем ошибку
            
            total_loss += loss.item()  # Суммируем ошибку
    
    average_loss = total_loss / num_batches  # Вычисляем среднюю ошибку
    return average_loss
