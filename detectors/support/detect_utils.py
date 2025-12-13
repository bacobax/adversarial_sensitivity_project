import torch


def get_device():
    """Get the best available device (CUDA or CPU).
    
    Returns:
        torch.device: Device to use for computation
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
