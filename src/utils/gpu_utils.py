import torch

def assign_device(gpu_id: int) -> torch.device:
    """Get the best available device with GPU ID."""
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    elif torch.backends.mps.is_available():
        return torch.device(f'mps')
    else:
        return torch.device('cpu')
