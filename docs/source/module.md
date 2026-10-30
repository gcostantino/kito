# KitoModule

A `KitoModule` holds the model, optimizer and loss, and defines what happens for one batch. The Engine runs the loops.

## Writing a module

Subclass `KitoModule` and implement two methods:

```python
import torch
import torch.nn as nn
from kito import KitoModule

class MyModel(KitoModule):
    def build_inner_model(self):
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        self.model_input_size = (1, 28, 28)   # without batch dimension

    def bind_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
```

The constructor takes only the config: `MyModel(config)`. With a config it reads `learning_rate`, `batch_size` and `module_name` from it, and builds the loss from `config.model.loss`.

## Lifecycle

`build()` calls `build_inner_model()` and marks the module as built. `associate_optimizer()` raises `RuntimeError` unless `build()` ran first, then calls `bind_optimizer()`. The properties `is_built`, `is_optimizer_set` and `is_weights_loaded` report this state.

## Per-batch methods

| Method | What it does |
|---|---|
| `training_step(batch, pbar_handler=None)` | Unpacks `(inputs, targets)`, moves them to the device, zeroes gradients, forward, loss, `backward()`, `optimizer.step()`. Returns `{'loss': loss}`. |
| `validation_step(batch, pbar_handler=None)` | Same without the update. Returns `loss`, `outputs`, `targets` and `inputs`. |
| `prediction_step(batch, pbar_handler=None)` | Takes the input from a tuple/list (first element), a tensor, or a dict (`'input'`, then `'data'`). Returns the model output after `handle_model_outputs`. |
| `pass_data_through_model(data)` | `self.model(data)`. |
| `compute_loss(data_pair, y_pred)` | Moves `data_pair[1]` to the device and calls `apply_loss`. |
| `apply_loss(y_pred, y_true)` | `self.loss(y_pred, y_true)`. |
| `send_data_to_device(data)` | Moves a tensor (or a nested dict/list/tuple of tensors) to `self.device`. |
| `handle_model_outputs(outputs)` | Hook; returns the outputs unchanged. |

Override any of these to change behavior. The optimizer update currently happens inside `training_step`.

## Other methods

- `load_weights(weight_path, strict=True)` and `save_weights(weight_path)`.
- `get_sample_input()` returns a random tensor of shape `(1, *model_input_size)` on the module's device.
- `summary(summary_depth=3)` prints a `torchsummary` model summary. It needs `model_input_size` and a built model.
- `_check_data_shape(batch)` is a hook called by the Engine on the first training batch; the default does nothing.
- `set_model_input_size(...)` raises `NotImplementedError` unless you override it.
