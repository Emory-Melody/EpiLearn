Installation
============

This section provides instructions on how to install the project either from source or from PyPI. Please prepare a terminal to run the following commands.

First, Please download and install `Anaconda <https://www.anaconda.com/download/success>`_ so that you can use commands from conda.

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

1.2 From PyPI
--------------

To install the project from PyPI, use the following command:

.. code-block:: bash

   pip install epilearn


2 Install Mandatory Packages
-------------------------------------

EpiLearn also requires pytorch>=1.20, torch_geometric and torch_scatter. 

For CPU version, Please follow:

.. code-block:: bash

   pip install torch
   pip install torch_geometric 
   pip install torch_scatter


For GPU version, please install correct versions that match the cuda version on your machine.
For more information, please refer to `Pytorch <https://pytorch.org/>`_, `PyG <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_ and `torch_scatter <https://pytorch-geometric.com/whl/torch-1.5.0.html>`_.
