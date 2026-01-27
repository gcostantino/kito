import torch


def assign_device(device_type: str = "cuda", gpu_id: int = 0) -> torch.device:
    """
    Assign device based on preference and availability.

    Args:
        device_type: Preferred device ("cuda", "mps", or "cpu")
        gpu_id: GPU ID for CUDA

    Returns:
        torch.device: Best available device

    Behavior:
        - "cuda": Use CUDA if available, else CPU (with warning)
        - "mps": Use MPS if available, else CPU (with warning)
        - "cpu": Always CPU (no fallback)
    """
    valid_devices = {"cuda", "mps", "cpu"}  # already valitated before, this check might be removed
    device_type_lower = device_type.lower()

    if device_type_lower not in valid_devices:
        raise ValueError(
            f"Invalid device_type: '{device_type}'. "
            f"Must be one of {valid_devices}."
        )

    if device_type_lower == "cuda":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{gpu_id}")
        else:
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")

    elif device_type_lower == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("Warning: MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")

    elif device_type_lower == "cpu":
        return torch.device("cpu")

def get_available_devices() -> dict:
    """
    Get information about all available devices.

    Returns:
        dict: Dictionary with device availability information

    Example:
        >>> info = get_available_devices()
        >>> print(info)
        {
            'cuda': True,
            'cuda_count': 2,
            'mps': False,
            'cpu': True
        }
    """
    return {
        'cuda': torch.cuda.is_available(),
        'cuda_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'mps': torch.backends.mps.is_available(),
        'cpu': True  # CPU is always available
    }


def validate_device_type(device_type: str, raise_error: bool = True) -> bool:
    """
    Validate device_type string.

    Args:
        device_type: Device type to validate
        raise_error: If True, raise ValueError on invalid type

    Returns:
        bool: True if valid, False otherwise

    Raises:
        ValueError: If device_type is invalid and raise_error=True
    """
    valid_devices = {"cuda", "mps", "cpu"}
    is_valid = device_type.lower() in valid_devices

    if not is_valid and raise_error:
        raise ValueError(
            f"Invalid device_type: '{device_type}'. "
            f"Must be one of {valid_devices}."
        )

    return is_valid