import torch

def function02(dataset):
    num_features = dataset.shape[1]
    weights = torch.rand(num_features, dtype=torch.float32, requires_grad=True)
    return weights
