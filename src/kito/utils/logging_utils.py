import logging
import os
import sys

ROOT_LOGGER_NAME = 'kito'
_DEFAULT_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
_configured = False


def _current_rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    for var in ('RANK', 'KITO_WORKER_RANK'):  # torchrun / Kito spawn, before dist init
        if var in os.environ:
            return int(os.environ[var])
    return 0


class _RankFilter(logging.Filter):
    """Rank 0 logs everything; other ranks only log errors."""

    def filter(self, record):
        return record.levelno >= logging.ERROR or _current_rank() == 0


def configure_logging(level=logging.INFO, stream=None, fmt=_DEFAULT_FORMAT,
                      propagate=False, force=False):
    """Configure the 'kito' logger. Safe to call more than once."""
    global _configured
    root = logging.getLogger(ROOT_LOGGER_NAME)
    if _configured and not force:
        if level is not None:
            root.setLevel(level)
        return root

    for h in list(root.handlers):
        if getattr(h, '_kito_handler', False):
            root.removeHandler(h)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler._kito_handler = True
    handler.setFormatter(logging.Formatter(fmt))
    handler.addFilter(_RankFilter())
    root.addHandler(handler)
    root.setLevel(level if level is not None else logging.INFO)
    root.propagate = propagate
    _configured = True
    return root


def get_logger(name: str = None) -> logging.Logger:
    """Return a logger in the 'kito' hierarchy. Use get_logger(__name__) in every module."""
    if not _configured:
        configure_logging()
    if not name or name == ROOT_LOGGER_NAME:
        return logging.getLogger(ROOT_LOGGER_NAME)
    if not name.startswith(ROOT_LOGGER_NAME + '.'):
        name = f'{ROOT_LOGGER_NAME}.{name}'
    return logging.getLogger(name)
