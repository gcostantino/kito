<p align="center">
  <img src="https://raw.githubusercontent.com/gcostantino/kito/refs/heads/main/assets/kito_simplelogo.png" alt="Kito Logo" width="400"/>
</p>

---
**Effortless PyTorch training and (scientific) experimentation - define your model setup, Kito handles the rest.**

[![Tests](https://github.com/gcostantino/kito/actions/workflows/tests.yml/badge.svg)](https://github.com/gcostantino/kito/actions)
[![PyPI version](https://img.shields.io/pypi/v/pytorch-kito)](https://pypi.org/project/pytorch-kito/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-kito)](https://pypi.org/project/pytorch-kito/)
[![Downloads](https://img.shields.io/pypi/dm/pytorch-kito?style=round-square)](https://pepy.tech/project/pytorch-kito)
[![Documentation Status](https://readthedocs.org/projects/kito/badge/?version=latest)](https://kito.readthedocs.io/en/latest/?badge=latest)

# Kito

Kito is a PyTorch training library thought for making (scientific) experimentation processes of deep learning developers easy. You write the model and a config; `Engine` runs the training and validation loops, checkpointing, logging, inference and distributed training (DDP).

- Documentation: https://kito.readthedocs.io
- Package: https://pypi.org/project/pytorch-kito/
- License: MIT

## Installation

```bash
pip install pytorch-kito
```

Requires Python 3.9 or newer. Optional extras: `dev` (tests and linting), `docs` (documentation build), `all`, and `tensorboard`, which also installs matplotlib for the image plotters.


## Minimal example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kito import Engine, KitoModule
from kito.config.moduleconfig import KitoModuleConfig, ModelConfig, TrainingConfig, WorkDirConfig


class MLP(KitoModule):
    def build_inner_model(self):
        self.model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
        self.model_input_size = (20,)

    def bind_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


config = KitoModuleConfig(
    training=TrainingConfig(learning_rate=1e-3, n_train_epochs=5, batch_size=32, train_mode=True),
    model=ModelConfig(module_name='mlp', input_data_size=(20,), loss='mse'),
    workdir=WorkDirConfig(work_directory='./outputs'),
)

x = torch.randn(512, 20)
y = x.sum(dim=1, keepdim=True)
train_loader = DataLoader(TensorDataset(x[:400], y[:400]), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(x[400:], y[400:]), batch_size=32)

engine = Engine(MLP(config), config)
engine.fit(train_loader, val_loader)
```

A module implements two methods: `build_inner_model` (set `self.model` and `self.model_input_size`) and `bind_optimizer` (set `self.optimizer`). The loss is looked up by name in the loss registry from `config.model.loss`. `fit` builds the model, creates the optimizer, moves the module to the device, runs `n_train_epochs` epochs (or `max_epochs` if given) and writes logs and the best checkpoint under `work_directory` (see [Output files](#output-files)).

## Inference

`fit` works only when `config.training.train_mode` is `True`; `predict` works only when it is `False`. Any other combination raises `RuntimeError`. Load trained weights through `config.model.weight_load_path`:

```python
import copy

infer_config = copy.deepcopy(config)
infer_config.training.train_mode = False
infer_config.model.weight_load_path = 'outputs/weights/mlp/best_mlp_<timestamp>_experiment.pt'

engine = Engine(MLP(infer_config), infer_config)
x_new = torch.randn(100, 20)
predictions = engine.predict(x_new)
print(predictions.shape)  # (100, 1)

```

`predict` accepts a DataLoader, a data pipeline (its test split), a Dataset, a NumPy array, a tensor, a list, or an `(x, y)` tuple of arrays. It returns a float32 NumPy array with one row per sample. With `config.model.save_inference_to_disk = True` and `config.model.inference_filename` set, the predictions are written to an HDF5 file (dataset `predictions`) and `predict` returns `None`. `prediction_step` must return a tensor.

## Configuration

Required: `TrainingConfig(learning_rate, n_train_epochs, batch_size, train_mode)` and `ModelConfig(module_name, input_data_size)`. Everything else has a default.

| Config | Used for | Main fields (default) |
|---|---|---|
| `ModelConfig` | model and loss | `loss` (registered loss name), `loss_dirs` (folders with custom losses), `train_codename` (`'experiment'`, part of output file names), `weight_load_path`, `save_inference_to_disk`, `inference_filename` |
| `DataConfig` | splits and loading | `train_ratio` (0.8), `val_ratio` (0.1; the test split gets the rest), `total_samples`, `shuffle_split` (`False`), `split_seed`, `num_workers`, `preprocessing` |
| `CallbacksConfig` | default callbacks | `enable_csv_logger`, `enable_text_logger`, `enable_model_checkpoint` (all `True`), `checkpoint_monitor` (`'val_loss'`), `enable_tensorboard` (`False`) |
| `WorkDirConfig` | output location | `work_directory` (`''`, which means the current directory) |

See the documentation for the full list of fields.

## Data and preprocessing

`GenericDataPipeline` wraps a dataset, applies preprocessing, splits it and builds the DataLoaders. Datasets return `(data, labels)`; `MemDataset(x, y)` holds arrays in memory and `H5Dataset(path)` reads the datasets `data` and `labels` from an HDF5 file lazily. By default the splits are contiguous slices in dataset order (`shuffle_split=True` uses a seeded permutation). An empty validation split is allowed: training then runs without validation.

```python
import numpy as np
from torch.utils.data import Subset

from kito import GenericDataPipeline, MemDataset
from kito.data.preprocessing import Standardization, compute_stats

dataset = MemDataset(x.numpy(), y.numpy())

# Statistics from the training part only (the first 80% with the default split)
train_part = Subset(dataset, range(int(len(dataset) * config.data.train_ratio)))
stats = compute_stats(train_part)

pipeline = GenericDataPipeline(
    config,
    dataset=dataset,
    preprocessing=[Standardization(stats['mean'], stats['std'])],
)
engine = Engine(MLP(config), config)
engine.fit(data_pipeline=pipeline)
```

Preprocessing steps work on one sample at a time and never estimate statistics from the sample they receive, so `mean`, `std` and similar values are always explicit. Use `compute_stats` to get them from the training data.

| Step | Parameters |
|---|---|
| `Standardization` | `mean`, `std`, `channel_axis=None` |
| `Normalize` | `data_min`, `data_max`, `min_val=0`, `max_val=1`, `channel_axis=None` |
| `Clip` | `min_value=None`, `max_value=None` |
| `LogTransform` | `offset=1.0`, `base='e'` (`'e'`, `'10'` or `'2'`) |
| `AddNoise` | `std=0.01`, `mean=0.0` (training only) |
| `ToTensor` | `dtype=float32`, `label_dtype=None` |

Steps change `data` only. Pass `apply_to='labels'` or `apply_to='both'` (all steps except `ToTensor`) to transform the targets too. Steps marked training only (`AddNoise`) are skipped for the validation and test splits and in `predict`. Steps can be given as instances, or in `config.data.preprocessing` as `PreprocessingStepConfig(name='standardization', params={'mean': 0.0, 'std': 1.0})`.

### Custom datasets

Subclass `KitoDataset`, implement `_load_sample` and `__len__`, and register it to select it from the config:

```python
import numpy as np

from kito import KitoDataset
from kito.data.registry import DATASETS


@DATASETS.register('noisy_line')
class NoisyLine(KitoDataset):
    def __init__(self, n):
        self.x = np.random.randn(n, 20).astype('float32')

    def _load_sample(self, index):
        return self.x[index], self.x[index].sum(keepdims=True)

    def __len__(self):
        return len(self.x)


config.data.dataset_type = 'noisy_line'
config.data.dataset_init_args = {'n': 500}
pipeline = GenericDataPipeline(config)
pipeline.setup()
```

## Callbacks

`fit` always creates the default callbacks enabled in `CallbacksConfig` and appends the ones you pass to it. To turn a default off, set its flag in `CallbacksConfig`:

```python
from kito.callbacks import Callback, EarlyStoppingCallback
from kito.config.moduleconfig import CallbacksConfig


class PrintLoss(Callback):
    def on_epoch_end(self, epoch, engine, model, logs=None, **kwargs):
        print(f"epoch {epoch}: {logs}")


config.callbacks = CallbacksConfig(enable_text_logger=False)
engine = Engine(MLP(config), config)
engine.fit(
    train_loader,
    val_loader,
    callbacks=[EarlyStoppingCallback(monitor='val_loss', patience=5), PrintLoss()],
)
```

Available hooks: `on_train_begin`, `on_epoch_begin`, `on_train_batch_begin`, `on_train_batch_end`, `on_validation_begin`, `on_validation_end`, `on_epoch_end`, `on_train_end`. Hooks receive keyword arguments, so accept `**kwargs`. `logs` in `on_epoch_end` contains `train_loss` and, when validation ran, `val_loss`. Setting `engine.stop_training = True` ends training after the current epoch.

Built in: `ModelCheckpoint`, `EarlyStoppingCallback`, `CSVLogger`, `TextLogger`, `TensorBoardScalars`, `TensorBoardHistograms`, `TensorBoardGraph`, `DDPAwareCallback`.

Without validation data there is no `val_loss`: set `checkpoint_monitor='train_loss'` in `CallbacksConfig` and monitor `train_loss` in `EarlyStoppingCallback`.

## Custom training step

Override `training_step` to change how a batch is processed. It must return a dict with the key `'loss'`:

```python
class ClippedMLP(MLP):
    def training_step(self, batch, pbar_handler=None):
        inputs, targets = (t.to(self.device) for t in batch)
        outputs = self.model(inputs)
        loss = self.compute_loss((inputs, targets), outputs)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return {'loss': loss}
```

`validation_step` and `prediction_step` can be overridden in the same way.

## Distributed training

Set `TrainingConfig(distributed_training=True)`. There are two ways to start:

- With `torchrun`: `torchrun --nproc_per_node=2 train.py`. The Engine reads the environment variables set by `torchrun` and joins the process group.
- Automatically: call `engine.fit(data_pipeline=pipeline)` in a plain `python train.py`. Kito spawns one process per GPU (override with the environment variable `KITO_NPROCS`; at least 2 are needed). You must pass a `data_pipeline`, not DataLoaders, and your module class must be constructible as `MyModule(config)`.

The data pipeline uses a `DistributedSampler` when the process group is already initialized as `setup()` runs. Create the `Engine` before calling `pipeline.setup()`, or let `fit` call it. Callbacks run on rank 0 only.

## Output files

Paths are relative to `work_directory`; `<timestamp>` has the form `31Oct2026-143000` and `<codename>` is `train_codename`.

| Callback | File |
|---|---|
| `CSVLogger` | `logs/csv/<module>_<timestamp>_<codename>.csv` |
| `TextLogger` | `logs/text/<module>_<timestamp>_<codename>.log` |
| `ModelCheckpoint` | `weights/<module>/best_<module>_<timestamp>_<codename>.pt` |
| TensorBoard callbacks (`enable_tensorboard=True`) | `logs/tensorboard/<module>/<timestamp>/<codename>/`, images in its `images/` subfolder |


## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md). Kito is released under the MIT license ([LICENSE](LICENSE)). Bugs and questions: https://github.com/gcostantino/kito/issues
