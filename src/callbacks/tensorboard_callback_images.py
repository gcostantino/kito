"""
Base and example callbacks for image plotting to TensorBoard.

The BaseImagePlotter provides a foundation for custom image plotting callbacks.
"""
from abc import abstractmethod
from typing import Optional, List

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from src.callbacks.callback_base import Callback


class BaseImagePlotter(Callback):
    """
    Base class for plotting images to TensorBoard.

    Subclasses must implement `create_figure()` to define custom plotting logic.

    Args:
        log_dir: Directory for TensorBoard logs
        tag: Tag for the image in TensorBoard
        freq: Frequency of plotting (every N epochs)
        batch_indices: Which batch indices to visualize (e.g., [0, 1, 2])

    Example:
        class MyCustomPlotter(BaseImagePlotter):
            def create_figure(self, val_data, val_outputs, epoch, **kwargs):
                fig, ax = plt.subplots()
                # Your custom plotting logic here
                ax.imshow(val_outputs[0].cpu().numpy())
                return fig

        plotter = MyCustomPlotter('logs/images', tag='predictions')
    """

    def __init__(
            self,
            log_dir: str,
            tag: str = 'validation_images',
            freq: int = 1,
            batch_indices: Optional[List[int]] = None
    ):
        self.log_dir = log_dir
        self.tag = tag
        self.freq = freq
        self.batch_indices = batch_indices or [0]
        self.writer = None

    def on_train_begin(self, engine, model, **kwargs):
        """Initialize TensorBoard writer."""
        self.writer = SummaryWriter(self.log_dir)

    @abstractmethod
    def create_figure(self, val_data, val_outputs, epoch, **kwargs):
        """
        Create matplotlib figure for visualization.

        Must be implemented by subclasses.

        Args:
            val_data: Validation data (tuple of inputs and targets)
            val_outputs: Model predictions on validation data
            epoch: Current epoch number
            **kwargs: Additional context from engine

        Returns:
            matplotlib.figure.Figure or list of figures
        """
        pass

    def on_epoch_end(self, epoch, engine, model, logs=None, **kwargs):
        """Plot images to TensorBoard."""
        if epoch % self.freq != 0:
            return

        # Get validation data and outputs from kwargs
        val_data = kwargs.get('val_data')
        val_outputs = kwargs.get('val_outputs')

        if val_data is None or val_outputs is None:
            return

        # Create figure(s)
        figures = self.create_figure(val_data, val_outputs, epoch, **kwargs)

        # Log to TensorBoard
        if isinstance(figures, (list, tuple)):
            # Multiple figures
            for i, fig in enumerate(figures):
                idx = self.batch_indices[i] if i < len(self.batch_indices) else i
                self.writer.add_figure(
                    f'{self.tag}_batch_{idx}',
                    fig,
                    global_step=epoch,
                    close=True
                )
        else:
            # Single figure
            self.writer.add_figure(
                self.tag,
                figures,
                global_step=epoch,
                close=True
            )

        self.writer.flush()

    def on_train_end(self, engine, model, **kwargs):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()


# ============================================================================
# EXAMPLE IMAGE PLOTTERS
# ============================================================================

class SimpleImagePlotter(BaseImagePlotter):
    """
    Simple image plotter for input/output comparison.

    Shows input, ground truth, and prediction side by side.

    Args:
        log_dir: Directory for TensorBoard logs
        tag: Tag for the image in TensorBoard
        freq: Frequency of plotting (every N epochs)
        batch_indices: Which batch indices to visualize
        cmap: Colormap for matplotlib (default: 'viridis')

    Example:
        plotter = SimpleImagePlotter(
            'logs/images',
            tag='validation',
            batch_indices=[0, 1, 2]
        )
    """

    def __init__(
            self,
            log_dir: str,
            tag: str = 'validation',
            freq: int = 1,
            batch_indices: Optional[List[int]] = None,
            cmap: str = 'viridis'
    ):
        super().__init__(log_dir, tag, freq, batch_indices)
        self.cmap = cmap

    def create_figure(self, val_data, val_outputs, epoch, **kwargs):
        """Create side-by-side comparison figure."""
        # Extract input and target
        val_input = val_data[0]  # (B, C, H, W) or (B, T, C, H, W)
        val_target = val_data[1]

        # Create figures for each batch index
        figures = []
        for idx in self.batch_indices:
            if idx >= val_input.shape[0]:
                break

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Get data for this batch index
            input_img = self._prepare_image(val_input[idx])
            target_img = self._prepare_image(val_target[idx])
            pred_img = self._prepare_image(val_outputs[idx])

            # Plot
            axes[0].imshow(input_img, cmap=self.cmap)
            axes[0].set_title('Input')
            axes[0].axis('off')

            axes[1].imshow(target_img, cmap=self.cmap)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')

            axes[2].imshow(pred_img, cmap=self.cmap)
            axes[2].set_title('Prediction')
            axes[2].axis('off')

            plt.tight_layout()
            figures.append(fig)

        return figures if len(figures) > 1 else figures[0]

    def _prepare_image(self, tensor):
        """
        Prepare tensor for visualization.

        Handles:
        - Time series: (T, C, H, W) -> take middle frame
        - Multi-channel: (C, H, W) -> take first channel
        - Single channel: (1, H, W) -> squeeze
        """
        # Move to CPU and detach
        img = tensor.detach().cpu()

        # Handle time series
        if img.ndim == 4:  # (T, C, H, W)
            img = img[img.shape[0] // 2]  # Take middle frame

        # Handle channels
        if img.ndim == 3:  # (C, H, W)
            if img.shape[0] == 1:
                img = img[0]  # Single channel
            elif img.shape[0] == 3:
                img = img.permute(1, 2, 0)  # RGB
            else:
                img = img[0]  # Take first channel

        return img.numpy()


class DifferencePlotter(BaseImagePlotter):
    """
    Plot input, prediction, and difference (error) map.

    Useful for regression tasks to visualize prediction errors.

    Args:
        log_dir: Directory for TensorBoard logs
        tag: Tag for the image in TensorBoard
        freq: Frequency of plotting (every N epochs)
        batch_indices: Which batch indices to visualize

    Example:
        plotter = DifferencePlotter('logs/images', tag='errors')
    """

    def __init__(
            self,
            log_dir: str,
            tag: str = 'difference',
            freq: int = 1,
            batch_indices: Optional[List[int]] = None
    ):
        super().__init__(log_dir, tag, freq, batch_indices)

    def create_figure(self, val_data, val_outputs, epoch, **kwargs):
        """Create figure with input, prediction, and difference."""
        val_input = val_data[0]
        val_target = val_data[1]

        figures = []
        for idx in self.batch_indices:
            if idx >= val_input.shape[0]:
                break

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            # Get data
            input_img = self._prepare_image(val_input[idx])
            target_img = self._prepare_image(val_target[idx])
            pred_img = self._prepare_image(val_outputs[idx])
            diff_img = target_img - pred_img

            # Plot
            axes[0].imshow(input_img, cmap='gray')
            axes[0].set_title('Input')
            axes[0].axis('off')

            axes[1].imshow(target_img, cmap='gray')
            axes[1].set_title('Target')
            axes[1].axis('off')

            axes[2].imshow(pred_img, cmap='gray')
            axes[2].set_title('Prediction')
            axes[2].axis('off')

            im = axes[3].imshow(diff_img, cmap='RdBu_r', vmin=-diff_img.std(), vmax=diff_img.std())
            axes[3].set_title('Difference (Target - Pred)')
            axes[3].axis('off')
            plt.colorbar(im, ax=axes[3])

            plt.tight_layout()
            figures.append(fig)

        return figures if len(figures) > 1 else figures[0]

    def _prepare_image(self, tensor):
        """Prepare tensor for visualization."""
        img = tensor.detach().cpu()

        # Handle time series
        if img.ndim == 4:
            img = img[img.shape[0] // 2]

        # Handle channels
        if img.ndim == 3:
            if img.shape[0] > 1:
                img = img[0]  # Take first channel
            else:
                img = img[0]

        return img.numpy()
