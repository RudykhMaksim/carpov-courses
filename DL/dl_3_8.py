def count_parameters_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool):
    # Параметры весов
    weight_params = in_channels * out_channels * (kernel_size * kernel_size)
    
    # Параметры смещения
    bias_params = out_channels if bias else 0
    
    # Общее количество параметров
    total_params = weight_params + bias_params
    
    return total_params
