import torchvision.models as models
import torch.nn as nn

def get_pretrained_model(model_name: str, num_classes: int, pretrained: bool = True):
    model_name = model_name.lower()

    if model_name == 'alexnet':
        model = models.alexnet(pretrained=pretrained)
        # Заменяем классификатор
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    
    elif model_name == 'vgg11':
        model = models.vgg11(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=pretrained, aux_logits=True)
        # Основной классификатор
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        # Дополнительные классификаторы (используются только во время обучения)
        if model.aux1:
            model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, num_classes)
        if model.aux2:
            model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_classes)
    
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model

