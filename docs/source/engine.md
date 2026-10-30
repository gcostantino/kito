# Engine: training

```python
from kito import Engine
engine = Engine(module, config)
```

`Engine` takes a `KitoModule` and the `KitoModuleConfig`; both are required. On construction it
picks the device, moves the module to it, and creates the logger, progress bars and a timestamp
(`%d%b%Y-%H%M%S`) used to name outputs.

## fit

```python
engine.fit(train_loader=None, val_loader=None, data_pipeline=None,
           max_epochs=None, callbacks=None)
```

`fit` is available only when `config.training.train_mode` is `True`. Give it either DataLoaders
or a `GenericDataPipeline`. With a pipeline, `setup()` runs if needed, and an empty validation
split is allowed: training then runs without validation and a warning is logged.

What `fit` does, in order:

1. Builds the model if `module.is_built` is false, and loads the pretrained weights first if
   `initialize_model_with_saved_weights` is `True`.
2. Creates the optimizer (`associate_optimizer()`) if none is set.
3. Checks the module and data loaders.
4. Uses `max_epochs` or, if `None`, `config.training.n_train_epochs`.
5. Wraps the model in `DistributedDataParallel` if `distributed_training` is `True`.
6. Creates the default callbacks (see Callbacks), appends the ones you pass, and calls the callbacks' setup.
7. Runs each epoch: training epoch, then validation epoch if a validation loader exists, then
   `on_epoch_end` with `logs`.

`logs` contains `train_loss`, and `val_loss` when validation ran. Callbacks that monitor
`val_loss` do not work without validation data. Use `monitor='train_loss'`.

Training stops after the epoch when any callback sets `engine.stop_training = True`. Ctrl-C
ends training cleanly. `on_train_end` always runs, and in DDP the model is unwrapped and the
process group destroyed afterwards.

## Weights

`engine.load_weights(weight_path=None, strict=True)` loads from `weight_path`, or from
`config.model.weight_load_path` when it is `None`. Failures are raised as `RuntimeError` with
the path and the original error.

## Default callbacks

`engine.get_default_callbacks()` returns the callbacks enabled by `CallbacksConfig`. Add
your own and pass the list to `fit`. If you pass callbacks to `fit` without calling
`get_default_callbacks()`, the defaults are still created and your callbacks are appended after them.

## What one epoch does

- **Training:** the model is set to `train()`. For each batch: `on_train_batch_begin`, `module.training_step(batch)`, `loss.item()`, `on_train_batch_end`. The epoch loss is the mean of the per-batch losses, averaged across ranks in DDP.
- **First batch:** `module._check_data_shape(batch)` is called once per `fit`.
- **Validation:** `eval()` and `torch.no_grad()`. `on_validation_begin`, one `validation_step` per batch, then `on_validation_end` with the mean loss.
- **Errors:** an empty loader raises `ValueError`.
- **Cost:** each batch calls `.item()` on the loss, which synchronizes the device once per batch.
