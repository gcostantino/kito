# Losses

The loss is chosen by name in `config.model.loss`. `KitoModule` resolves it once, when the module is created, and
the default training and validation steps call it as `loss(y_pred, y_true)`. If `config.model.loss` is empty,
`module.loss` is `None` and you must override `compute_loss` or `apply_loss` yourself.

## Choosing a loss

`config.model.loss` accepts a registered name or a dictionary. `kito.losses.get_loss` also passes through an
already-built `nn.Module`.

```python
from kito.config.moduleconfig import ModelConfig

ModelConfig(module_name='m', input_data_size=(20,), loss='mse')

# Parameters in a 'params' dictionary
ModelConfig(module_name='m', input_data_size=(20,),
            loss={'name': 'weighted_mse', 'params': {'weight': 2.0}})

# Parameters next to the name (equivalent)
ModelConfig(module_name='m', input_data_size=(20,),
            loss={'name': 'weighted_mse', 'weight': 2.0})
```

A dictionary without a `name` key raises `ValueError`. Names are case-insensitive. An unknown name raises
`ValueError` with the list of available losses, after searching the directories described under
[Custom losses](#custom-losses).

## Built-in names

Every PyTorch loss is registered under three kinds of names: the lowercased class name (`mseloss`), a short
name, and, for some, a verbose name such as `mean_squared_error`. Common short names:

| PyTorch class | Short names |
|---|---|
| `MSELoss` | `mse`, `l2` |
| `L1Loss` | `l1`, `mae` |
| `CrossEntropyLoss` | `cross_entropy`, `cross_entropy_loss` |
| `NLLLoss` | `nll` |
| `BCELoss` | `bce` |
| `BCEWithLogitsLoss` | `bce_with_logits` |
| `HuberLoss` | `huber` |
| `SmoothL1Loss` | `smooth_l1` |
| `KLDivLoss` | `kl_div` |
| `PoissonNLLLoss` | `poisson_nll` |
| `GaussianNLLLoss` | `gaussian_nll` |
| `CTCLoss` | `ctc` |

Other names exist for the margin, embedding and triplet losses (`margin_ranking`, `hinge_embedding`,
`soft_margin`, `multi_margin`, `triplet_margin`, `cosine_embedding`, and others). Print the complete list with:

```python
from kito.losses import LossRegistry

print(LossRegistry.list_available())
```

Names are not always the PyTorch function names: `mse_loss` is not registered, while `mse` is.

## Losses provided by Kito

| Name | Arguments | Description |
|---|---|---|
| `weighted_mse` | `weight=1.0`, `reduction='mean'` | MSE multiplied by a global weight |
| `combined` | `losses` | Weighted sum of other registered losses |
| `ssim` | `window_size=11`, `sigma=1.5`, `K1=0.01`, `K2=0.03`, `L=1.0` | Structural similarity loss for images |

`combined` takes a list of dictionaries with a `name`, an optional `weight` (default `1.0`) and optional `params`:

```python
ModelConfig(
    module_name='m',
    input_data_size=(20,),
    loss={'name': 'combined',
          'params': {'losses': [{'name': 'mse', 'weight': 1.0},
                                {'name': 'l1', 'weight': 0.5}]}},
)
```

## Custom losses

Register an `nn.Module` with `LossRegistry.register`. The class receives the `params` of the config as keyword
arguments, and `forward` takes the predictions and the targets.

```python
import torch
import torch.nn as nn

from kito.losses import LossRegistry


@LossRegistry.register('relative_l2')
class RelativeL2(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.norm(pred - target) / (torch.norm(target) + self.eps)
```

Select it like any other loss: `loss={'name': 'relative_l2', 'params': {'eps': 1e-6}}`.

Registration happens when the class is defined, so the class must be imported before the module is created.
Defining it in your training script is enough. Registering a different class under a name that is already taken
raises `ValueError`.

### Loading losses from files

Put each loss in its own `.py` file with the decorator above, and Kito imports the files for you:

- Directories in `config.model.loss_dirs` are searched when the module is created.
- If a name is not registered, Kito also searches `./losses`, `./custom_losses` and `./lib/losses`, relative to
  the working directory.
- Files whose names start with `_` are skipped, and each file is imported on its own, so a package
  `__init__.py` is not needed.
- A file that fails to import produces a warning and is skipped.

To load a directory yourself, call `kito.losses.discover_losses(directory, verbose=True)`. It returns the
number of newly registered losses.

```python
ModelConfig(module_name='m', input_data_size=(20,), loss='relative_l2', loss_dirs=['./my_losses'])
```

## Custom loss computation

For losses that need more than `(prediction, target)`, such as several outputs or the current epoch, override one
of these methods in your module:

- `compute_loss(self, data_pair, y_pred, **kwargs)`: receives the `(inputs, targets)` batch and moves the targets
  to the module's device before calling `apply_loss`.
- `apply_loss(self, y_pred, y_true, **kwargs)`: calls `self.loss(y_pred, y_true)` by default.

```python
class WeightedMLP(MLP):
    def apply_loss(self, y_pred, y_true, **kwargs):
        return self.loss(y_pred, y_true) + 1e-3 * y_pred.abs().mean()
```

The old `kito.utils.loss_utils.get_loss` is deprecated. Use `kito.losses.get_loss` or the config.
