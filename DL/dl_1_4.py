import torch
import torch.nn as nn

def function04(x: torch.Tensor, y: torch.Tensor):
    n_features = x.shape[1]
    model = nn.Linear(n_features, 1, bias=False)

    # Обучение модели
    num_epochs = 1000  # Количество эпох для обучения
    lr = 1e-2  # Шаг обучения

    for _ in range(num_epochs):
        # Прямой проход
        y_pred = model(x)
        
        # Вычисление функции потерь
        loss = ((y_pred.squeeze() - y) ** 2).mean()

        # Обратное распространение
        model.zero_grad()
        loss.backward()
        
        # Обновление весов
        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad
        
        # Проверка на MSE
        if loss.item() < 0.3:
            break

    return model
