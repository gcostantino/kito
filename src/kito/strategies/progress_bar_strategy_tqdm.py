import time
from collections import deque

from tqdm import tqdm

from kito.strategies.progress_bar_strategy import BaseProgressBarHandler


class TqdmProgressBarHandler(BaseProgressBarHandler):
    """
    Standard progress bar using tqdm with time per step tracking.

    Features:
    - Smoothed time per step (averaged over last 20 steps)
    - Clean bar format with elapsed/remaining time
    - Automatic metric display from training loop
    """
    _header_printed = False

    def __init__(self):
        self.pbar = None
        self.total = 0
        self.current = 0
        self.step_start_time = None
        self.step_times = deque(maxlen=20)  # Track last 20 steps for smooth average

    @staticmethod
    def _colored(text, color_code):
        """Helper to colorize text."""
        return f"\033[{color_code}m{text}\033[0m"

    def init(self, total, verbosity_level, message=''):
        """
        Initialize progress bar.

        Args:
            total: Total number of steps
            verbosity_level: Display level (0 = silent, >0 = show bar)
            message: Description message for the bar
        """
        self.total = total
        self.current = 0
        self.step_start_time = time.time()
        self.step_times.clear()

        if verbosity_level > 0:
            # Print header using tqdm's write with colors
            if not TqdmProgressBarHandler._header_printed:
                # Calculate spacing to match tqdm's layout
                desc_width = len(message) + 10  # Description + percentage space
                bar_width = 50  # Default bar width (can adjust)

                # Build aligned header
                header = (
                    f"{'# completed epochs':<{desc_width}}"  # Left-aligned description area
                    f"|{'Progress Bar':^{bar_width}}|"  # Centered bar area
                    f" # completed batches [time elapsed > remaining] Metrics"  # Right side info
                )

                tqdm.write("\n" + self._colored("=" * 130, "1;31"))
                tqdm.write(self._colored(header, "31"))
                tqdm.write(self._colored("=" * 130, "1;31") + "\n")
                TqdmProgressBarHandler._header_printed = True

            self.pbar = tqdm(
                total=total,
                desc=message,
                ncols=130,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}'
            )
        else:
            self.pbar = None

    def step(self, current_step, values=None):
        """
        Update progress bar.

        Args:
            current_step: Current step number
            values: List of (key, value) tuples to display (e.g., [("loss", 0.123)])
        """
        if self.pbar is not None:
            # Calculate time for this step
            current_time = time.time()
            if self.step_start_time is not None:
                step_time = 1e03 * (current_time - self.step_start_time)
                self.step_times.append(step_time)
            self.step_start_time = current_time

            # Update progress
            increment = current_step - self.current
            self.current = current_step

            # Build postfix with metrics
            postfix = {}
            if values:
                for key, val in values:
                    postfix[key.strip().rstrip(':')] = f'{val:.4f}'

            # Add time per step (average of last N steps)
            if self.step_times:
                avg_step_time = sum(self.step_times) / len(self.step_times)
                postfix['ms/step'] = f'{avg_step_time:.3f}'

            self.pbar.set_postfix(postfix, refresh=True)
            self.pbar.update(increment)

    def close(self):
        """Close and cleanup progress bar."""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class DDPTqdmProgressBarHandler(TqdmProgressBarHandler):
    """
    DDP-aware progress bar using tqdm.

    Only displays progress bar on the master process (rank 0).
    All other processes run silently.
    """

    def __init__(self):
        super().__init__()
        import torch.distributed as dist

        # Check if we're the master process
        self.is_master = (
                dist.is_available() and
                dist.is_initialized() and
                dist.get_rank() == 0
        )

    def init(self, total, verbosity_level, message=''):
        """Initialize progress bar only on master process."""
        if self.is_master:
            super().init(total, verbosity_level, message)
        else:
            self.pbar = None

    def step(self, current_step, values=None):
        """Update progress bar only on master process."""
        if self.is_master:
            super().step(current_step, values)

    def close(self):
        """Close progress bar only on master process."""
        if self.is_master:
            super().close()
