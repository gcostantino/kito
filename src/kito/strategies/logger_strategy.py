import logging
import os

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
        self.logger = logging.getLogger('kito')
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
        self.logger.propagate = False

    def log_info(self, msg):
        self.logger.info(msg)

    def log_warning(self, msg):
        self.logger.warning(msg)

    def log_error(self, msg):
        self.logger.error(msg)


class DDPLogger(DefaultLogger):
    def __init__(self, log_level=logging.INFO):
        super().__init__(log_level)

    @property
    def is_driver(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return int(os.environ.get('RANK', 0)) == 0  # before init, rely on launcher env

    def log_info(self, msg):
        if self.is_driver:
            super().log_info(msg)

    def log_warning(self, msg):
        if self.is_driver:
            super().log_warning(msg)

    def log_error(self, msg):
        if self.is_driver:
            super().log_error(msg)
