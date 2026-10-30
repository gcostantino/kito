# Configuration

Kito is configured with plain Python dataclasses from `kito.config.moduleconfig`.
`KitoModuleConfig` groups five sections:

```python
from kito.config.moduleconfig import (
    KitoModuleConfig, TrainingConfig, ModelConfig, WorkDirConfig, DataConfig, CallbacksConfig,
)

config = KitoModuleConfig(
    training=TrainingConfig(learning_rate=1e-3, n_train_epochs=20, batch_size=32, train_mode=True),
    model=ModelConfig(module_name="MyModel", input_data_size=(1, 28, 28), loss="mse"),
)
```

`training` and `model` are required. `workdir`, `data` and `callbacks` have defaults.

## TrainingConfig

| Field | Default | Meaning |
|---|---|---|
| `learning_rate` | required | Stored on the module as `learning_rate`. |
| `n_train_epochs` | required | Number of epochs, used when `fit` is not given `max_epochs`. |
| `batch_size` | required | Stored on the module as `batch_size`. |
| `train_mode` | required | `True` for training, `False` for inference. |
| `train_verbosity_level`, `val_verbosity_level`, `test_verbosity_level` | `2` | Progress display: 0 silent, 1 progress bar, 2 detailed. |
| `distributed_training` | `False` | Enables DDP. |
| `device_id` | `0` | GPU index when not distributed. |
| `ddp_find_unused_parameters` | `False` | Passed to `DistributedDataParallel(find_unused_parameters=...)`. |
| `initialize_model_with_saved_weights` | `False` | Load weights before training. |
| `device_type` | `"cuda"` | `"cuda"`, `"mps"` or `"cpu"`. Case-insensitive; anything else raises `ValueError`. |

## ModelConfig

| Field | Default | Meaning |
|---|---|---|
| `module_name` | required | Name of the module, used in output file names. |
| `input_data_size` | required | Input shape without the batch dimension. |
| `loss` | `{}` | A loss name, a dict, or an instantiated loss. See "Loss definition". |
| `loss_dirs` | `[]` | Directories with custom loss files to load before building the loss. |
| `train_codename` | `"experiment"` | Label used in log, checkpoint and TensorBoard names. |
| `weight_load_path` | `""` | Weights file to load. |
| `save_inference_to_disk` | `False` | Write predictions to HDF5. |
| `inference_filename` | `""` | Output file for predictions. |
| `tensorboard_img_id` | `"training_viz"` | Tag for the default image plotter. |
| `batch_idx_viz` | `[0]` | See "Known duplicates". |

### Loss definition

1. `"mse"`
2. `{"name": "mse"}`
3. `{"name": "weighted_mse", "params": {"weight": 2.0}}`
4. `{"name": "weighted_mse", "weight": 2.0}`

## DataConfig

| Field | Default | Meaning |
|---|---|---|
| `dataset_type` | `"h5dataset"` | Registered dataset name (`"h5dataset"`, `"memdataset"` or a custom one). |
| `dataset_path` | `""` | Path to the data file. |
| `dataset_init_args` | `{}` | Keyword arguments for the dataset constructor. Takes precedence over `dataset_path`. |
| `load_into_memory` | `False` | Load the whole dataset into RAM. |
| `train_ratio`, `val_ratio` | `0.8`, `0.1` | Split fractions. The test fraction is the remainder. |
| `total_samples` | `None` | Use only this many samples (`None` = all). |
| `shuffle_split` | `False` | Random, seeded split instead of contiguous slices. |
| `split_seed` | `0` | Seed for the shuffled split. |
| `preprocessing` | `[]` | List of `PreprocessingStepConfig(name, params)`, applied in order. |
| `num_workers`, `prefetch_factor`, `pin_memory`, `persistent_workers` | `0`, `2`, `False`, `False` | DataLoader settings. |

## CallbacksConfig

Controls the default callbacks the Engine creates.

| Field | Default |
|---|---|
| `enable_csv_logger`, `enable_text_logger`, `enable_model_checkpoint` | `True` |
| `checkpoint_monitor`, `checkpoint_mode` | `"val_loss"`, `"min"` |
| `checkpoint_save_best_only`, `checkpoint_verbose` | `True`, `False` |
| `enable_tensorboard` (master switch) | `False` |
| `tensorboard_scalars`, `tensorboard_histograms`, `tensorboard_graph` | `True` |
| `tensorboard_histogram_freq` | `5` |
| `tensorboard_images`, `tensorboard_image_freq` | `False`, `1` |
| `tensorboard_batch_indices` | `[0]` |
| `image_plotter_class` | `None` (auto-detect) |

To use your own callbacks, take the defaults and append: `callbacks = engine.get_default_callbacks(); callbacks.append(MyCallback()); engine.fit(..., callbacks=callbacks)`.

## Known duplicates

`ModelConfig.batch_idx_viz` and `CallbacksConfig.tensorboard_batch_indices` overlap. The default callback creation reads `tensorboard_batch_indices` and `tensorboard_img_id`.
