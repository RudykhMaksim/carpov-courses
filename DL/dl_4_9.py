import torch
import torch.nn as nn
from torch.utils.data import DataLoader

@torch.no_grad()
def predict_tta(model: nn.Module, loader: DataLoader, device: torch.device, iterations: int = 2):
    """
    Performs Test Time Augmentation (TTA) predictions using the given model and dataloader.
    
    Args:
        model (nn.Module): The neural network model
        loader (DataLoader): DataLoader with test data
        device (torch.device): Device to run the model on
        iterations (int): Number of TTA iterations
    
    Returns:
        torch.Tensor: Predicted classes for each sample
    """
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    # Initialize list to store predictions from all iterations
    all_predictions = []
    
    # Get the number of samples in the dataset
    n_samples = len(loader.dataset)
    # Get number of classes from first batch
    first_batch = next(iter(loader))
    n_classes = model(first_batch[0].to(device)).shape[1]
    
    for _ in range(iterations):
        # Initialize tensor to store predictions for current iteration
        iteration_predictions = []
        
        # Iterate through the dataloader
        for batch in loader:
            # Get inputs and move to device
            inputs = batch[0].to(device)
            
            # Get model outputs (logits)
            outputs = model(inputs)
            iteration_predictions.append(outputs)
        
        # Stack predictions for current iteration
        iteration_predictions = torch.cat(iteration_predictions, dim=0)
        all_predictions.append(iteration_predictions)
    
    # Stack all iteration predictions into [N, C, iterations]
    all_predictions = torch.stack(all_predictions, dim=2)
    
    # Average predictions across iterations
    mean_predictions = torch.mean(all_predictions, dim=2)
    
    # Get predicted classes
    predicted_classes = torch.argmax(mean_predictions, dim=1)
    
    return predicted_classes
