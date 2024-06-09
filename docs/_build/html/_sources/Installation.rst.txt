Installation
============

This section provides instructions on how to install the project either from source or from PyPI.

From Source
-----------

To install the project from source, follow these steps:

.. code-block:: bash

   git clone https://github.com/Emory-Melody/EpiLearn.git
   cd EpiLearn

   conda create -n epilearn python=3.9
   conda activate epilearn

   python setup.py install
   pip install pytorch_geometric

From PyPI
---------

To install the project from PyPI, use the following command:

.. code-block:: bash

   pip install epilearn
