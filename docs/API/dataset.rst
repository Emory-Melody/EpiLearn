Dataset
===================================

In EpiLearn, we use **UniversalDataset** to load preprocessed datasets. For customized data, we can simply initialize the UniversalDataset given features, graphs, and states.


UniversalDataset
--------------------

.. autoclass:: epilearn.data.dataset.UniversalDataset
    :members:


Preprocessed Datasets
===================================

We collect epidemic data from various sources including the followings:


**Temporal Data**

   * `Tycho_v1.0.0 <https://www.tycho.pitt.edu/data/>`_: Including eight diseases collected across 50 US states and 122 US cities from 1916 to 2009.
   * `Measles <https://github.com/msylau/measles_competing_risks/tree/master>`_: Contains measles infections in England and Wales across 954 urban centers (cities and towns) from 1944 to 1964.

**Spatial&Temporal Data**

   * **Covid_static**: Contains covid infections with static graph. `[1] <https://github.com/littlecherry-art/DASTGN/tree/master>`_
   * **Covid_dynamic**: Contains covid infections with dynamic graph. `[2] <https://github.com/HySonLab/pandemic_tgnn/tree/main>`_ `[3] <https://github.com/deepkashiwa20/MepoGNN/tree/main>`_

**Dataset Loading**

Loading Measle and Tycho Datasets:

.. code-block:: python

    from epilearn.data import UniversalDataset  

    tycho_dataset = UniversalDataset(name='Tycho_v1', root='./tmp/')    

    measle_dataset = UniversalDataset(name='Measles', root='./tmp/')    


For covid data, we support the Dataset from Johns Hopkings University:

.. code-block:: python

    from epilearn.data import UniversalDataset

    jhu_dataset = UniversalDataset(name='JHU_covid', root='./tmp/')


For other countries, please use 'Covid\_'+'country' to acquire the correspnding covid dataset. Currently, we support countries like China, Brazil, Austria, England, France, Italy, Newzealand, and Spain.

.. code-block:: python

    from epilearn.data import UniversalDataset

    covid_dataset = UniversalDataset(name='Covid_Brazil', root='./tmp/')


