import torch

def function02(dataset):
    num_features = dataset.shape[1]
    weights = torch.rand(num_features, dtype=torch.float32, requires_grad=True)
    return weights

def function03(x: torch.Tensor, y: torch.Tensor):
    # Инициализируем веса
    weights = function02(x)
    
    learning_rate = 1e-2
    num_iterations = 1000  # Количество итераций для градиентного спуска

    for _ in range(num_iterations):
        # Прогнозирование
        y_pred = x @ weights
        
        # Вычисление ошибки
        loss = torch.mean((y_pred - y) ** 2)  # MSE
        
        # Обнуление градиентов
        weights.grad = None
        
        # Обратное распространение
        loss.backward()
        
        # Обновление весов
        with torch.no_grad():
            weights -= learning_rate * weights.grad

    return weights
