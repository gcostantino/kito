Quickstart
==========

This page trains a small regression model on random data and then uses it for inference. The blocks build on
each other and must be run in order.

Training
--------

A module implements ``build_inner_model`` (set ``self.model`` and ``self.model_input_size``) and
``bind_optimizer`` (set ``self.optimizer``). The loss is chosen by name in the config.

.. code-block:: python

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

Logs and the best checkpoint are written under ``./outputs``:

- ``logs/csv/<module>_<timestamp>_<codename>.csv``
- ``logs/text/<module>_<timestamp>_<codename>.log``
- ``weights/<module>/best_<module>_<timestamp>_<codename>.pt``

Inference
---------

``predict`` works only when ``train_mode`` is ``False``. Point ``weight_load_path`` at the checkpoint written
above.

.. code-block:: python

   config.training.train_mode = False
   config.model.weight_load_path = 'outputs/weights/mlp/best_mlp_<timestamp>_experiment.pt'

   engine = Engine(MLP(config), config)
   predictions = engine.predict(torch.randn(100, 20))
   print(predictions.shape)  # (100, 1)

Next steps
----------

- :doc:`module`: what a ``KitoModule`` provides and which methods you can override.
- :doc:`data`: datasets, splits and preprocessing.
- :doc:`callbacks`: default and custom callbacks.
- :doc:`configuration`: all configuration classes.
