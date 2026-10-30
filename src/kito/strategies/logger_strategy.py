from kito.utils.logging_utils import configure_logging, get_logger


class BaseLogger:
    def log_info(self, msg):
        raise NotImplementedError

    def log_warning(self, msg):
        raise NotImplementedError

    def log_error(self, msg):
        raise NotImplementedError


class DefaultLogger(BaseLogger):
    def __init__(self, log_level=None, name='engine'):
        if log_level is not None:
            configure_logging(level=log_level)
        self.logger = get_logger(name)

    def log_info(self, msg):
        self.logger.info(msg)

    def log_warning(self, msg):
        self.logger.warning(msg)

    def log_error(self, msg):
        self.logger.error(msg)


class DDPLogger(DefaultLogger):
    """Kept for backwards compatibility or future development: rank filtering now lives in the 'kito' handler."""
