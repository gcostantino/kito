from torch.nn.parallel import DataParallel, DistributedDataParallel


def unwrap_model(model):
    """Return the underlying model (strips DDP / DataParallel wrappers)."""
    while isinstance(model, (DistributedDataParallel, DataParallel)):
        model = model.module
    return model
