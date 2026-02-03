import logging

import torch.distributed as dist


class BaseLogger:
    def log_info(self, msg):
        raise NotImplementedError

    def log_warning(self, msg):
        raise NotImplementedError

    def log_error(self, msg):
        raise NotImplementedError


class DefaultLogger(BaseLogger):
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)

    def log_info(self, msg):
        self.logger.info(msg)

    def log_warning(self, msg):
        self.logger.warning(msg)

    def log_error(self, msg):
        self.logger.error(msg)


class DDPLogger(DefaultLogger):
    def __init__(self, log_level=logging.INFO):
        super().__init__(log_level)
        # DON'T check rank here - do it lazily when first needed
        self._is_driver = None

    @property
    def is_driver(self):
        """Lazy evaluation of rank (only checked when first used)."""
        if self._is_driver is None:
            # Check if DDP is initialized
            if dist.is_available() and dist.is_initialized():
                self._is_driver = dist.get_rank() == 0
            else:
                # Fallback: assume driver if DDP not initialized
                self._is_driver = True
        return self._is_driver

    def log_info(self, msg):
        if self.is_driver:
            super().log_info(msg)

    def log_warning(self, msg):
        if self.is_driver:
            super().log_warning(msg)

    def log_error(self, msg):
        if self.is_driver:
            super().log_error(msg)
