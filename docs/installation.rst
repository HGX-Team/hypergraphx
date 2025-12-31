⚙️ Installation
===============

You can install HGX using pip:

.. code:: bash

   pip install hypergraphx

.. note::

   If you are using notebooks, install the docs requirements too to enable rich outputs.

To install from source:

.. code:: bash

   pip install -e .

Optional dependencies
---------------------

If you only need core functionality, the base install is enough. For docs or notebooks:

.. code:: bash

   pip install -r requirements/docs.txt
   pip install .[docs]

Check the repository ``requirements/runtime.txt`` for the required runtime dependencies.
Development and documentation dependencies live in ``requirements/dev.txt`` and
``requirements/docs.txt`` respectively.
You can also install optional extras with ``pip install .[dev]`` or ``pip install .[docs]``.
