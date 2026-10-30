# Data

## Datasets

Datasets return `(data, labels)` pairs and subclass `kito.data.datasets.KitoDataset`
(implement `_load_sample(index)` and `__len__`).

| Class | Registered name | Constructor | Notes |
|---|---|---|---|
| `H5Dataset` | `h5dataset` | `H5Dataset(path)` | Reads datasets named `data` and `labels` from an HDF5 file, opened lazily. The file handle is not pickled, so it works with `num_workers > 0`. Wrong path or missing keys raise `RuntimeError`. |
| `MemDataset` | `memdataset` | `MemDataset(x, y)` | Arrays or tensors held in memory. `len(x)` and `len(y)` must match, otherwise `ValueError`. |
| `InferenceDataset` | not registered | `InferenceDataset(x, y=None)` | Used by `predict` for arrays; float64 input is converted to float32. Labels optional. |

## Preprocessing

Steps are applied **per sample**, in `__getitem__`, on `(data, labels)`. They are registered under these names:
`normalize`, `standardization`, `clip_outliers`, `detrend`, `add_noise`, `log_transform`, `to_tensor`, `pipeline`.
`PreprocessedDataset(base_dataset, steps)` accepts `PreprocessingStepConfig(name, params)`, dicts
`{'name': ..., 'params': ...}`, and `Preprocessing` instances, and chains them in a `Pipeline`.

## GenericDataPipeline

```python
pipeline = GenericDataPipeline(config, dataset=my_dataset, preprocessing=[...])
pipeline.setup()
train_loader = pipeline.train_dataloader()
```

`setup()` wraps the dataset in `PreprocessedDataset` if steps are given, splits it, and creates a
DataLoader for every non-empty split.

**Splits.** With `total_samples` (default: all), `train_size = int(total * train_ratio)` and
`val_size = int(total * val_ratio)`. The test split gets what remains of the first `total` samples. By default the
splits are contiguous slices in dataset order. With `shuffle_split=True`, a permutation seeded by `split_seed` is used.
Ratios must each be in [0, 1] and sum to at most 1; `total_samples` larger than the dataset raises `ValueError`.

**Loaders.** The training loader shuffles; validation and test loaders do not. `drop_last` is not set,
so the last batch can be smaller. `batch_size` comes from `config.training`; `num_workers`, `pin_memory`,
`persistent_workers` and `prefetch_factor` from `config.data`.

**Empty splits.** An empty split gets no loader. Asking for it raises `EmptySplitError` (a
`ValueError` subclass). `Engine.fit` treats an empty validation split as "no validation".

**DDP.** If distributed training is enabled and the process group is already initialized when
`setup()` runs, each split uses a `DistributedSampler`. Create the `Engine` first, or let
`fit(data_pipeline=...)` call `setup()`.

## Registries

`DATASETS` and `PREPROCESSING` map names to classes. Register with `@DATASETS.register('name')`,
instantiate with `DATASETS.create('name', **kwargs)`, list with `list_registered()`.

## Preprocessing steps

A step takes one sample and returns `(data, labels)`. Steps change `data` only, unless you pass
`apply_to='labels'` or `apply_to='both'`. Statistics are never estimated from the sample being
processed; pass them explicitly.

| Name | Class | Parameters | What it does |
|---|---|---|---|
| `standardization` | `Standardization` | `mean`, `std`, `channel_axis=None` | `(x - mean) / std`; has `inverse()` |
| `normalize` | `Normalize` | `data_min`, `data_max`, `min_val=0`, `max_val=1`, `channel_axis=None` | Linear rescale; has `inverse()` |
| `clip` | `Clip` | `min_value=None`, `max_value=None` | Clips to the bounds |
| `log_transform` | `LogTransform` | `offset=1.0`, `base='e'` (`'e'`, `'10'`, `'2'`) | `log(x + offset)`; has `inverse()` |
| `add_noise` | `AddNoise` | `std=0.01`, `mean=0.0` | Gaussian noise; training only |
| `to_tensor` | `ToTensor` | `dtype=float32`, `label_dtype=None` | Converts to tensors |
| `pipeline` | `Pipeline` | `steps` | Applies steps in order |

Get the statistics from the training data:

```python
from kito.data.preprocessing import compute_stats, Standardization
stats = compute_stats(raw_train_dataset, channel_axis=0)
step = Standardization(stats['mean'], stats['std'], channel_axis=0)
```

Steps marked training-only are skipped for validation and test data, and by `predict`.
To add your own, subclass `Preprocessing`, implement `__call__(data, labels)`, and register it with
`@PREPROCESSING.register('name')`.

