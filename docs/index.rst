.. yun_exe documentation master file, created by
   sphinx-quickstart on Tue May 28 15:59:56 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Enpowering Epidemic Modeling with EpiLearn!
============================================

Epilearn is a python library built for epidemic modeling, providing abundant tools to quickly construct and test epidemic models as well as analyze epidemic data.

It covers four tasks -- forecasting, nowcasting, scenario modeling and outbreak detection -- behind
one model interface, and evaluates them with a shared rolling-window protocol that produces
conformal prediction intervals. The model zoo has 65 classes (:doc:`API/models`), 45 of which are
wired into the config-driven benchmark. Start with the :doc:`Quickstart`, then use the
:doc:`Benchmark` page to compare many models from a single YAML config.

.. toctree::
   :maxdepth: 1
   :caption: Installation

   Installation

.. toctree::
   :maxdepth: 1
   :caption: Usage

   Quickstart
   Benchmark

   tutorials/task_building
   tutorials/simulation
   tutorials/utils
   tutorials/customization

.. toctree::
   :maxdepth: 1
   :caption: Package API

   API/dataset
   API/models
   API/tasks
   API/utils
   API/visualization
