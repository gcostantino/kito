# Engine: prediction

`predict` is available only when `config.training.train_mode` is `False`.

```text
engine.predict(data=None, *, batch_size=None, preprocessing=None,
               test_loader=None, data_pipeline=None)
```

`data` can be a DataLoader, a Dataset, an array, a tensor or a pipeline. `test_loader` and `data_pipeline` are older keywords that are used when `data` is `None`.
Data is converted to a DataLoader with `as_dataloader`. `batch_size` and `preprocessing` apply
to that conversion.

Before predicting, the Engine builds the model if needed. If the module has learnable weights
and none are loaded, it loads `config.model.weight_load_path`, or warns when that is empty. Then it runs
`module.prediction_step` on each batch with gradients disabled and the model in `eval()` mode.

Predictions are kept in memory, or written to an HDF5 file when `config.model.save_inference_to_disk` is
`True`. In that case `config.model.inference_filename` must be set, otherwise `predict` raises
`ValueError`.

## Accepted input

`predict(data)` accepts: a `DataLoader` (used as is), a data pipeline (its test split), a `Dataset`, a
NumPy array, a tensor, a list, or an `(x, y)` tuple of arrays. Arrays and tuples are wrapped in
`InferenceDataset` and loaded with `num_workers=0`; `shuffle=False`, `drop_last=False`.

`config.data.preprocessing` is applied to datasets and arrays by default. Pass
`preprocessing=False` for none, or a `Preprocessing` instance or a list of steps to replace the config's.
Steps marked `training_only` are dropped. DataLoaders and pipelines are assumed ready and are not preprocessed.

## Output

- `prediction_step` must return a `torch.Tensor`, otherwise `TypeError`.
- In memory: a NumPy `float32` array with one row per sample.
- On disk: an HDF5 file, dataset name `predictions`, `float32`, chunked with at most 32 samples per chunk. `predict` returns `None`.
- If the module sets `standard_data_shape`, it defines the per-sample output shape; otherwise the shape is taken from the first prediction.
- The first batch is run once to find the output shape and then again in the main loop.
