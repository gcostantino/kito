API reference
=============

Engine and module
-----------------

.. autoclass:: kito.Engine
   :members:

.. autoclass:: kito.KitoModule
   :members:

Configuration
-------------

.. autoclass:: kito.config.moduleconfig.KitoModuleConfig
   :members:
   :undoc-members:

.. autoclass:: kito.config.moduleconfig.TrainingConfig
   :members:
   :undoc-members:

.. autoclass:: kito.config.moduleconfig.ModelConfig
   :members:
   :undoc-members:

.. autoclass:: kito.config.moduleconfig.DataConfig
   :members:
   :undoc-members:

.. autoclass:: kito.config.moduleconfig.CallbacksConfig
   :members:
   :undoc-members:

.. autoclass:: kito.config.moduleconfig.WorkDirConfig
   :members:
   :undoc-members:

.. autoclass:: kito.config.moduleconfig.PreprocessingStepConfig
   :members:
   :undoc-members:

Data
----

.. autoclass:: kito.GenericDataPipeline
   :members:

.. autoclass:: kito.KitoDataset
   :members:

.. autoclass:: kito.MemDataset

.. autoclass:: kito.H5Dataset

Preprocessing
-------------

.. autoclass:: kito.data.preprocessing.Preprocessing
   :members:

.. autoclass:: kito.data.preprocessing.Pipeline

.. autoclass:: kito.data.preprocessing.Standardization
   :members: inverse

.. autoclass:: kito.data.preprocessing.Normalize
   :members: inverse

.. autoclass:: kito.data.preprocessing.Clip

.. autoclass:: kito.data.preprocessing.LogTransform
   :members: inverse

.. autoclass:: kito.data.preprocessing.AddNoise

.. autoclass:: kito.data.preprocessing.ToTensor

.. autofunction:: kito.data.preprocessing.compute_stats

Callbacks
---------

.. autoclass:: kito.callbacks.Callback
   :members:

.. autoclass:: kito.callbacks.ModelCheckpoint

.. autoclass:: kito.callbacks.EarlyStoppingCallback

.. autoclass:: kito.callbacks.CSVLogger

.. autoclass:: kito.callbacks.TextLogger

.. autoclass:: kito.callbacks.TensorBoardScalars

.. autoclass:: kito.callbacks.TensorBoardHistograms

.. autoclass:: kito.callbacks.TensorBoardGraph

.. autoclass:: kito.callbacks.DDPAwareCallback
