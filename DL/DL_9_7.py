import torch
import torch.nn as nn

class Similarity1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoder_states: torch.Tensor, decoder_state: torch.Tensor):
        # Убедимся, что размеры совпадают
        assert encoder_states.dim() == 2, "encoder_states must be [T, N]"
        assert decoder_state.dim() == 1, "decoder_state must be [N]"
        
        # Вычисляем скалярное произведение для каждого состояния энкодера с состоянием декодера
        similarity = torch.matmul(encoder_states, decoder_state)
        
        return similarity
