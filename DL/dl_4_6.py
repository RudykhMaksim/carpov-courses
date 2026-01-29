import torchvision.transforms as T

def get_augmentations(train: bool = True) -> T.Compose:
    if train:
        return T.Compose([
            T.Resize((224, 224)),  # Изменение размера
            T.RandomHorizontalFlip(),  # Случайное горизонтальное отражение
            T.RandomCrop(224, padding=4),  # Случайная обрезка с паддингом
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Изменение цвета
            T.ToTensor(),  # Преобразование в тензор
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # Нормализация для CIFAR10
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),  # Изменение размера
            T.ToTensor(),  # Преобразование в тензор
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # Нормализация для CIFAR10
        ])

