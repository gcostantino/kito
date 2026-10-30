# Callbacks

Callbacks subclass `kito.callbacks.callback_base.Callback`. All hooks are optional and receive
keyword arguments; accept `**kwargs`.

| Hook | When | Extra arguments |
|---|---|---|
| `setup(engine, **context)` | Once, before training | `timestamp`, `work_directory`, `module_name`, `train_codename` |
| `on_train_begin(engine, model)` | Start of `fit` | |
| `on_epoch_begin(epoch, engine, model)` | Start of each epoch (1-based) | |
| `on_train_batch_begin(batch, engine, model)` | Before each training batch (`batch` is the 0-based index) | `batch_data` |
| `on_train_batch_end(batch, engine, model, logs)` | After each training batch | `logs={'loss': float}` |
| `on_validation_begin(epoch, engine, model)` | Before the validation loop | |
| `on_validation_end(epoch, engine, model, logs)` | After the validation loop | `logs={'val_loss': float}` |
| `on_epoch_end(epoch, engine, model, logs)` | End of each epoch | `val_data`, `val_outputs` |
| `on_train_end(engine, model)` | End of `fit`, also after Ctrl-C or an error | |

`logs` in `on_epoch_end` has `train_loss`, and `val_loss` if validation ran.
`val_data` is `(inputs, targets)` and `val_outputs` the outputs of the **last validation batch only**.
Set `engine.stop_training = True` to stop after the current epoch.

## Default callbacks

`fit` creates these from `CallbacksConfig` and runs your callbacks after them:

| Callback | Enabled by default | Output file |
|---|---|---|
| `CSVLogger` | yes | `<work_directory>/logs/csv/<module>_<timestamp>_<codename>.csv` |
| `TextLogger` | yes | `<work_directory>/logs/text/<module>_<timestamp>_<codename>.log` |
| `ModelCheckpoint` | yes | `<work_directory>/weights/<module>/best_<module>_<timestamp>_<codename>.pt` |
| `TensorBoardScalars`, `TensorBoardHistograms`, `TensorBoardGraph` | only if `enable_tensorboard` | `<work_directory>/logs/tensorboard/<module>/<timestamp>/<codename>` |
| Image plotter | only if `enable_tensorboard` and `tensorboard_images` | `.../images` under the TensorBoard directory |

If `work_directory` is empty, these paths are relative to the current directory.
`$VARIABLES` in `work_directory` are expanded.

The image plotter class is, in order: `CallbacksConfig.image_plotter_class`, the module's
`get_preferred_image_plotter()` if it defines one, otherwise `SimpleImagePlotter`. It receives
`ModelConfig.tensorboard_img_id` as its tag, plus `tensorboard_image_freq` and `tensorboard_batch_indices`.

## Built-in callbacks

- `ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='min', verbose=False)`
- `EarlyStoppingCallback(monitor='val_loss', patience=10, mode='min', min_delta=0.0)`
- `CSVLogger(filename, separator=',', append=False)`
- `DDPAwareCallback(callback, master_rank=0)`: runs the wrapped callback only on that rank.
  In DDP the Engine wraps every callback in it.

Callbacks that monitor `val_loss` find nothing when there is no validation split. Use `monitor='train_loss'` there.

## Files written by the built-in callbacks

- **CSVLogger:** the header is `epoch` plus the keys of `logs` from the first epoch (`train_loss`, and `val_loss` when
  validating). One row per epoch, flushed after each. Default mode overwrites the file; `append=True` appends.
- **TextLogger:** one line per epoch, `Epoch N: key=value, ...` with four decimals.
- **ModelCheckpoint:** saves `state_dict()` of the unwrapped model (so DDP weights load into a plain model). The file is
  written to `<path>.tmp` and then renamed, so an interrupted run never leaves a half-written checkpoint. `filepath`
  may contain `{epoch}` and metric names such as `{val_loss}`, filled in via `str.format`.
  With `save_best_only=True` it saves only when the monitored value improves; the best value resets at each `fit`.
- **EarlyStoppingCallback:** counts epochs without improvement (better than best by more than `min_delta`); at
  `patience` it sets `engine.stop_training = True`. If `monitor` is not in `logs`, it warns once and never triggers.
