Installation
============

Requirements
------------

- Python 3.9 or newer
- PyTorch 2.0 or newer

From PyPI
---------

.. code-block:: bash

   pip install pytorch-kito

Optional extras: ``tensorboard`` (TensorBoard callbacks), ``dev`` (tests and linting), ``docs`` (building this
documentation) and ``all``:

.. code-block:: bash

   pip install "pytorch-kito[tensorboard]"

From source
-----------

.. code-block:: bash

   git clone https://github.com/gcostantino/kito.git
   cd kito
   pip install -e ".[dev]"
