Installation
============

This section provides instructions on how to install the project either from source or from PyPI. Please prepare a terminal to run the following commands.

EpiLearn requires **Python >= 3.9**. First, Please download and install `Anaconda <https://www.anaconda.com/download/success>`_ so that you can use commands from conda.

1.1 From Source
-----------------

To install the project from source, follow these steps:

.. code-block:: bash

   git clone https://github.com/Emory-Melody/EpiLearn.git
   cd EpiLearn

   # create a new environment using conda
   conda create -n epilearn python=3.9

   # activate environment
   conda activate epilearn

   # install the package and its dependencies
   pip install .

The build is driven by ``pyproject.toml``; there is no ``setup.py`` anymore, so the
old ``python setup.py install`` command no longer works. Use ``pip install .`` (or
``pip install -e .`` for an editable checkout) instead.

1.2 From PyPI
--------------

To install the project from PyPI, use the following command:

.. code-block:: bash

   pip install epilearn

.. note::

   The wheel contains the ``epilearn`` package only. ``configs/``, ``datasets/``,
   ``examples/`` and ``tests/`` are **not** installed, so the YAML benchmark runs on
   :doc:`Benchmark` (``--config configs/...``) and the examples that read
   ``datasets/toy_features.csv`` or ``examples/example.pt`` need a source checkout.
   ``pip install epilearn`` plus the ``git clone`` above is the usual combination.
   ``Dataset().load_toy_dataset()`` is the exception: with no local
   ``./datasets/features.npy`` it downloads ~11 MB into a ``./datasets/`` directory
   under your current working directory, so it works from a PyPI install as long as
   the machine has network access.


2 Install Mandatory Packages
-------------------------------------

Both commands above install every required dependency for you, including
``torch>=1.13`` and ``torch_geometric>=2.3``. Note that **torch_geometric is a hard
requirement**, not an optional one: several ``SpatialTemporal`` models import it at
package import time, so ``import epilearn`` fails without it.

If you prefer to install the dependencies yourself, or you need a specific
PyTorch build, install them before EpiLearn.

For the CPU version, Please follow:

.. code-block:: bash

   pip install torch
   pip install torch_geometric

For the GPU version, please install correct versions that match the cuda version on your machine.
For more information, please refer to `Pytorch <https://pytorch.org/>`_ and `PyG <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_.

The remaining requirements are ordinary scientific-Python packages -- numpy, scipy,
pandas, matplotlib, seaborn, networkx, scikit-learn, statsmodels, optuna, psutil,
tqdm, einops, fastdtw and PyYAML -- and are listed in ``requirements.txt``.


3 Optional Extras
-------------------

Two groups of dependencies are optional, because they are only needed by a subset
of the library:

.. code-block:: bash

   # foundation-model backends: Chronos, Moirai (uni2ts), Moment (momentfm), TimesFM
   pip install epilearn[chronos]      # also [moirai], [moment], [timesfm]

   # interactive plots: Forecast.plot_preds(..., backend='plotly', interactive=True)
   pip install epilearn[plot]

.. warning::

   The plotly backend of ``plot_preds`` still uses the ``titlefont`` layout property,
   which plotly removed in 6.0, so it needs a 5.x release. Until that is fixed,
   install ``pip install "plotly<6"`` rather than relying on the ``plot`` extra's
   ``plotly>=5.0`` range. The default matplotlib backend is unaffected.

The foundation-model backends are imported lazily, inside each model's
``_load_model()``. Without the ``foundation`` extra the rest of EpiLearn works
normally: the benchmark reports those models as skipped, and using one directly
raises an ``ImportError`` telling you what to install.

If you installed from source, the same extras are available as
``pip install ".[chronos]"`` (or ``[moirai]`` / ``[moment]`` / ``[timesfm]``) and ``pip install ".[plot]"``.


4 Verify the Installation
---------------------------

.. code-block:: bash

   python -c "import epilearn; print(epilearn.__version__)"

.. code-block:: text

   0.1.0

Installing the package also adds the benchmark command, which is equivalent to
``python -m epilearn.benchmark``:

.. code-block:: bash

   epilearn-benchmark --help

.. code-block:: text

   usage: epilearn-benchmark [-h] --config CONFIG [--output OUTPUT]

   EpiLearn Benchmark Runner

   options:
     -h, --help            show this help message and exit
     --config CONFIG, -c CONFIG
                           Path to config YAML file
     --output OUTPUT, -o OUTPUT
                           Output directory (overrides config)
