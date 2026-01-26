import torch

'''def assign_device(gpu_id: int) -> torch.device:
    """Get the best available device with GPU ID."""
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    elif torch.backends.mps.is_available():
        return torch.device(f'mps')
    else:
        return torch.device('cpu')'''

def assign_device(gpu_id: int, logger=None) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        # LOG: "Using CUDA device cuda:0"
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        # LOG: "CUDA unavailable. Using Apple MPS device"
        # LOG WARNING: "MPS doesn't support multi-GPU"
    else:
        device = torch.device('cpu')
        # LOG WARNING: "No GPU detected. Using CPU (training will be slower)"
    return device