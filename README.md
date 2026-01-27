# Kito

**Effortless PyTorch training - define your model, Kito handles the rest.**

[![Tests](https://github.com/gcostantino/kito/actions/workflows/tests.yml/badge.svg)](https://github.com/gcostantino/kito/actions)
[![PyPI version](https://img.shields.io/pypi/v/pytorch-kito)](https://pypi.org/project/pytorch-kito/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Downloads](https://img.shields.io/pypi/dm/pytorch-kito)](https://pypi.org/project/pytorch-kito/)

Kito is a lightweight PyTorch training library that eliminates boilerplate code. Define your model architecture and loss function - Kito automatically handles training loops, optimization, callbacks, distributed training, and more.

## ✨ Key Features

- **Zero Boilerplate** - No training loops, no optimizer setup, no device management
- **Auto-Everything** - Automatic model building, optimizer binding, and device assignment
- **Built-in DDP** - Distributed training works out of the box
- **Smart Callbacks** - TensorBoard, checkpointing, logging, and custom callbacks
- **Flexible** - Simple for beginners, powerful for experts
- **Lightweight** - Minimal dependencies, pure PyTorch under the hood

## Quick Start

### Installation

```bash
pip install pytorch-kito
```

### Your First Model in 3 Steps

```python
import torch.nn as nn
from kito import Engine, KitoModule

# 1. Define your model
class MyModel(KitoModule):
    def build_inner_model(self):
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.model_input_size = (784,)

    def bind_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

# 2. Initialize
model = MyModel('MyModel', device, config)
engine = Engine(model, config)

# 3. Train! (That's it - everything else is automatic)
engine.fit(train_loader, val_loader, max_epochs=10)
```

## Philosophy

Kito follows a **"define once, train anywhere"** philosophy:

1. **You focus on**: Model architecture and research ideas
2. **Kito handles**: Training loops, optimization, distributed training, callbacks

Perfect for researchers who want to iterate quickly without rewriting training code.

## Core Concepts

### KitoModule

Your model inherits from `KitoModule` and implements two methods:

```python
class MyModel(KitoModule):
    def build_inner_model(self):
        # Define your architecture
        self.model = nn.Sequential(...)
        self.model_input_size = (C, H, W)  # Input shape

    def bind_optimizer(self):
        # Choose your optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
```

### Engine

The `Engine` orchestrates everything:

```python
engine = Engine(module, config)

# Training
engine.fit(train_loader, val_loader, max_epochs=100)

# Inference
predictions = engine.predict(test_loader)
```

### Data Pipeline

Kito provides a clean data pipeline with preprocessing:

```python
from kito.data import H5Dataset, GenericDataPipeline
from kito.data.preprocessing import Pipeline, Normalize, ToTensor

# Create dataset
dataset = H5Dataset('data.h5')

# Add preprocessing
preprocessing = Pipeline([
    Normalize(min_val=0.0, max_val=1.0),
    ToTensor()
])

# Setup data pipeline
pipeline = GenericDataPipeline(
    config=config,
    dataset=dataset,
    preprocessing=preprocessing
)
pipeline.setup()

# Get dataloaders
train_loader = pipeline.train_dataloader()
val_loader = pipeline.val_dataloader()
```

### Callbacks

Kito includes powerful callbacks for common tasks:

```python
from kito.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

callbacks = [
    ModelCheckpoint('best_model.pt', monitor='val_loss', mode='min'),
    EarlyStopping(patience=10, monitor='val_loss'),
    CSVLogger('training.csv')
]

engine.fit(train_loader, val_loader, callbacks=callbacks)
```

Or create custom callbacks:

```python
from kito.callbacks import Callback

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs, **kwargs):
        print(f"Epoch {epoch}: loss={logs['train_loss']:.4f}")
```

## Advanced Features

### Distributed Training (DDP)

Enable distributed training with one config change:

```python
config.training.distributed_training = True

# Everything else stays the same!
engine.fit(train_loader, val_loader, max_epochs=100)
```

### Custom Training Logic

Override `training_step` for custom behavior:

```python
class MyModel(KitoModule):
    def training_step(self, batch, pbar_handler=None):
        inputs, targets = batch

        # Custom forward pass
        outputs = self.model(inputs)
        loss = self.compute_loss((inputs, targets), outputs)

        # Custom backward (e.g., gradient clipping)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {'loss': loss}
```

### Multiple Datasets

Kito supports HDF5 and in-memory datasets out of the box:

```python
from kito.data import H5Dataset, MemDataset

# HDF5 dataset (lazy loading)
dataset = H5Dataset('large_data.h5')

# In-memory dataset (fast)
dataset = MemDataset(x_train, y_train)
```

Register custom datasets easily:

```python
from kito.data.registry import DATASETS

@DATASETS.register('my_custom_dataset')
class MyDataset(KitoDataset):
    def _load_sample(self, index):
        return data, labels
```

## 📦 Installation Options

```bash
# Basic installation
pip install pytorch-kito

# With TensorBoard support
pip install pytorch-kito[tensorboard]

# Development installation
pip install pytorch-kito[dev]

# Everything
pip install pytorch-kito[all]
```

## 🤝 Contributing

Contributions are very welcome! Please check out our [Contributing Guide](CONTRIBUTING.md).

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Kito is inspired by PyTorch Lightning and Keras, aiming to bring similar ease-of-use to pure PyTorch workflows for researchers.

## Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/gcostantino/kito/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/gcostantino/kito/discussions)

---

**Made with ❤️ for the PyTorch community**
