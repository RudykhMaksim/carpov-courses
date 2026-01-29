import torch
from torch import nn
from torch.utils.data import DataLoader

def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()  # Установка модели в режим оценки
    predictions = []  # Список для хранения предсказаний
    
    with torch.no_grad():  # Отключение градиентов
        for inputs, _ in loader:  # Загружаем данные из DataLoader
            inputs = inputs.to(device)  # Перемещаем данные на устройство
            outputs = model(inputs)  # Проводим forward pass
            _, preds = torch.max(outputs, 1)  # Получаем классы с максимальным логитом
            predictions.append(preds.cpu())  # Добавляем предсказания в список и перемещаем на CPU

    return torch.cat(predictions)  # Конкатенация всех предсказаний
