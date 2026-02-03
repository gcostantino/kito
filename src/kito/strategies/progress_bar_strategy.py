from abc import ABC

import torch


class BaseProgressBarHandler(ABC):

    def init(self, n_target_elements, verbosity_level, message=None):
        pass

    def step(self, i, values):
        pass

    def finalize(self, i, values):
        pass


class StandardProgressBarHandler(BaseProgressBarHandler):
    def __init__(self):
        self.progress_bar = None
        # print("Epoch {}/{}".format(epoch + 1, self.n_train_epochs))  # might be replaced by logger

    def init(self, n_target_elements, verbosity_level, message=None):
        import pkbar
        if message is not None:
            print(message)
        self.progress_bar = pkbar.Kbar(target=n_target_elements, always_stateful=False, width=25,
                                       verbose=verbosity_level)
        self.progress_bar.update(0, None)

    def step(self, i, values):
        self.progress_bar.update(i, values)

    def finalize(self, i, values):
        self.progress_bar.add(i, values)


class DDPProgressBarHandler(BaseProgressBarHandler):
    def __init__(self):
        self.progress_bar = None
        self.is_driver = (torch.distributed.get_rank() == 0)
        # print("Epoch {}/{}".format(epoch + 1, self.n_train_epochs))  # might be replaced by logger

    def init(self, n_target_elements, verbosity_level, message=None):
        import pkbar
        if self.is_driver:
            if message is not None:
                print(message)
            self.progress_bar = pkbar.Kbar(target=n_target_elements, always_stateful=False, width=25,
                                           verbose=verbosity_level)

    def step(self, i, values):
        if self.is_driver:
            self.progress_bar.update(i, values)

    def finalize(self, i, values):
        if self.is_driver:
            self.progress_bar.add(i, values)
