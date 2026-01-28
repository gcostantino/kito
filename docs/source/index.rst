Kito Documentation
==================

**Kito** is a PyTorch-based deep learning framework thought for researchers. Kito offers effortless training of
deep learning models, by automatically handling training loops, optimization, callbacks, distributed training, and more.

.. note::
   This documentation is under active development.

Quick Start
-----------

Installation::

   pip install pytorch-kito

Basic usage::

   from kito import Engine, KitoModule

   # Define your model
   class MyModel(KitoModule):
       def build_inner_model(self):
           self.model = nn.Sequential(...)

   # Train
   module = MyModel("model_name", device, config)
   engine = Engine(module, config)
   engine.fit(train_loader, val_loader)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   api

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/basic_usage

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
