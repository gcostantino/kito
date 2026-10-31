# TensorBoard

Kito can write scalars, parameter histograms, the model graph and validation images to TensorBoard. It is off by
default. Enable it with the master switch in `CallbacksConfig`:

```python
from kito.config.moduleconfig import CallbacksConfig

config.callbacks = CallbacksConfig(enable_tensorboard=True)
```

View the logs with `tensorboard --logdir <work_directory>/logs/tensorboard`.

## Options

| Field | Default | Effect |
|---|---|---|
| `enable_tensorboard` | `False` | Master switch; nothing below applies without it |
| `tensorboard_scalars` | `True` | Log every key of the epoch `logs` (`train_loss`, `val_loss`) |
| `tensorboard_histograms` | `True` | Log a histogram of every parameter |
| `tensorboard_histogram_freq` | `5` | Log histograms every N epochs |
| `tensorboard_graph` | `True` | Log the model graph once, at the start of training |
| `tensorboard_images` | `False` | Log figures made by an image plotter |
| `tensorboard_image_freq` | `1` | Log images every N epochs |
| `tensorboard_batch_indices` | `[0]` | Which samples of the validation batch are plotted |
| `image_plotter_class` | `None` | Plotter class to use (see [Image plotters](#image-plotters)) |

Epochs are numbered from 1. Histograms are written when the epoch number is divisible by the frequency, so with
`tensorboard_histogram_freq=5` the first histograms appear at epoch 5.

## Where the files go

Each run writes to its own directory:

```text
<work_directory>/logs/tensorboard/<module_name>/<timestamp>/<train_codename>/
<work_directory>/logs/tensorboard/<module_name>/<timestamp>/<train_codename>/images/
```

Scalars, histograms and the graph share the first directory. Images go into the `images` subdirectory, which
TensorBoard shows as a separate run.

In a test run with `tensorboard_histogram_freq=2` and 4 epochs, the first directory contained `train_loss` and
`val_loss` at epochs 1 to 4, parameter histograms at epochs 2 and 4, and the graph. The `images` directory
contained one image tag, `training_viz`, at epochs 1 to 4.

## What is logged

| Callback | Logs | Tag names |
|---|---|---|
| `TensorBoardScalars` | Every key of the epoch `logs` | `train_loss`, `val_loss` |
| `TensorBoardHistograms` | Every parameter of the model | The parameter names, such as `weight` |
| `TensorBoardGraph` | The model graph, once | n/a |

The graph needs a sample input. With `enable_tensorboard`, the Engine takes it from the module's
`get_sample_input()`. If no sample input is available, the graph is skipped without a warning.

## Image plotters

Images need `matplotlib` (`pip install "pytorch-kito[tensorboard]"`). Set `tensorboard_images=True` to enable
them. The Engine picks the plotter class in this order:

1. `config.callbacks.image_plotter_class`, if set.
2. `module.get_preferred_image_plotter()`, if the module defines it.
3. `SimpleImagePlotter`.

Available classes in `kito.callbacks.tensorboard_callback_images`:

| Class | Extra arguments | Tag when built directly |
|---|---|---|
| `SimpleImagePlotter` | `cmap='viridis'` | `validation` |
| `DifferencePlotter` | none beyond the base arguments | `difference` |
| `BaseImagePlotter` | n/a (subclass it) | from `_get_default_tag()` |

The Engine creates the plotter with tag=config.model.tensorboard_img_id (default "training_viz").
Set that field to change the tag. A plotter’s own default tag applies only when you construct it yourself.
### Custom plotter

Subclass `BaseImagePlotter`, implement `create_figure`, and return one matplotlib figure or a list of figures.
`SimpleImagePlotter.create_figure` is the best template for how `val_data` and `val_outputs` are laid out.

```python
from kito.callbacks.tensorboard_callback_images import BaseImagePlotter


class ErrorPlotter(BaseImagePlotter):
    def create_figure(self, val_data, val_outputs, epoch, **kwargs):
        ...  # build and return a matplotlib figure


config.callbacks = CallbacksConfig(
    enable_tensorboard=True,
    tensorboard_images=True,
    image_plotter_class=ErrorPlotter,
)
config.model.tensorboard_img_id = 'abs_error'
# fit...
```

The constructor takes `log_dir=None, tag=None, freq=1, batch_indices=None`. `log_dir` is filled in by the Engine.

## Using the callbacks directly

The callbacks can also be passed to `fit` without the config switch:

```python
from kito.callbacks import TensorBoardScalars, TensorBoardHistograms, TensorBoardGraph

engine.fit(
    train_loader,
    val_loader,
    callbacks=[
        TensorBoardScalars('runs/exp1'),
        TensorBoardHistograms('runs/exp1', freq=10),
        TensorBoardGraph('runs/exp1', input_to_model=lambda: torch.randn(1, 20)),
    ],
)
```

`TensorBoardHistograms` defaults to `freq=1`. `TensorBoardGraph` without `input_to_model` falls back to
`engine.get_sample_input` and does nothing if that doesn't exist.
