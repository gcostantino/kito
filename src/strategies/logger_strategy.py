import logging

import torch


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
        # Only log from rank 0
        self.is_driver = torch.distributed.get_rank() == 0

    def log_info(self, msg):
        if self.is_driver:
            super().log_info(msg)

    def log_warning(self, msg):
        if self.is_driver:
            super().log_warning(msg)

    def log_error(self, msg):
        if self.is_driver:
            super().log_error(msg)
