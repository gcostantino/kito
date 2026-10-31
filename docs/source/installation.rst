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

Optional extras: `dev` (tests and linting), `docs` (documentation build), `all`, and `tensorboard`, which also installs matplotlib for the image plotters.


.. code-block:: bash

   pip install "pytorch-kito[tensorboard]"

From source
-----------

.. code-block:: bash

   git clone https://github.com/gcostantino/kito.git
   cd kito
   pip install -e ".[dev]"
