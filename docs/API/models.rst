Models
===================================

Spatial Models
===================================


GAT
----------

.. autoclass:: epilearn.models.Spatial.GAT.GAT
    :members:


GCN 
----------

.. autoclass:: epilearn.models.Spatial.GCN.GCN
    :members:

GIN 
----------

.. autoclass:: epilearn.models.Spatial.GIN.GIN
    :members:

SAGE 
----------

.. autoclass:: epilearn.models.Spatial.SAGE.SAGE
    :members:

Code Example
-------------

.. code-block:: python

    import torch
    from epilearn.models.Spatial import GCN

    num_features = 4
    num_classes = 2
    lookback = 1 # inputs size
    horizon = 2 # predicts size

    graph = torch.round(torch.rand((47,47)))
    features = torch.round(torch.rand((10,47,1,4))) 
    node_target = torch.round(torch.rand((10,47))) 

    model=GCN(num_features=num_features, num_classes=horizon, device='cpu')
    model.fit(
            train_input=features, 
            train_target=node_target, 
            train_graph=graph,
            val_input=None,
            val_target=None,
            val_graph=None,
            epochs=20,
            loss='ce'
            )


Temporal Models
===================================


ARIMA
----------

.. autoclass:: epilearn.models.Temporal.ARIMA.VARMAXModel
    :members:


DLINEAR
----------

.. autoclass:: epilearn.models.Temporal.Dlinear.DlinearModel
    :members:


GRU
----------

.. autoclass:: epilearn.models.Temporal.GRU.GRUModel
    :members:


LSTM
----------

.. autoclass:: epilearn.models.Temporal.LSTM.LSTMModel
    :members:


SIR
----------

.. autoclass:: epilearn.models.Temporal.SIR.SIR
    :members:


SIS
----------

.. autoclass:: epilearn.models.Temporal.SIR.SIS
    :members:

SEIR
----------

.. autoclass:: epilearn.models.Temporal.SIR.SEIR
    :members:

XGBOOST
----------

.. autoclass:: epilearn.models.Temporal.XGB.XGBModel
    :members:

Code Example
-------------

.. code-block:: python

    import torch
    from epilearn.models.Temporal.GRU import GRUModel

    num_features = 1
    lookback = 16 # inputs size
    horizon = 3 # predicts size

    features = torch.round(torch.rand((10, lookback, num_features)))
    node_target = torch.round(torch.rand((10, horizon, num_features))) 

    model=GRUModel(num_features=num_features, num_timesteps_input=lookback, num_timesteps_output=horizon, device='cpu')
    model.fit(
            train_input=features, 
            train_target=node_target, 
            val_input=None,
            val_target=None,
            val_graph=None,
            epochs=20,
            loss='mse'
            )

Spatial-Temporal Models
===================================


ATMGNN
----------

.. autoclass:: epilearn.models.SpatialTemporal.ATMGNN.ATMGNN
    :members:

MPNN_LSTM
--------------------

.. autoclass:: epilearn.models.SpatialTemporal.ATMGNN.MPNN_LSTM
    :members:

CNNRNN_Res
--------------------

.. autoclass:: epilearn.models.SpatialTemporal.CNNRNN_Res.CNNRNN_Res
    :members:


ColaGNN
--------------------

.. autoclass:: epilearn.models.SpatialTemporal.ColaGNN.ColaGNN
    :members:

DASTGN
----------

.. autoclass:: epilearn.models.SpatialTemporal.DASTGN.DASTGN
    :members:


DCRNN
----------

.. autoclass:: epilearn.models.SpatialTemporal.DCRNN.DCRNN
    :members:



DMP
----------

.. autoclass:: epilearn.models.SpatialTemporal.DMP.DMP
    :members:


EpiColaGNN
--------------------

.. autoclass:: epilearn.models.SpatialTemporal.EpiColaGNN.EpiColaGNN
    :members:


EpiGNN
----------

.. autoclass:: epilearn.models.SpatialTemporal.EpiGNN.EpiGNN
    :members:

GraphWaveNet
--------------------

.. autoclass:: epilearn.models.SpatialTemporal.GraphWaveNet.GraphWaveNet
    :members:




MepoGNN
----------

.. autoclass:: epilearn.models.SpatialTemporal.MepoGNN.MepoGNN
    :members:


NetSIR
----------

.. autoclass:: epilearn.models.SpatialTemporal.NetworkSIR.NetSIR
    :members:


STAN
----------

.. autoclass:: epilearn.models.SpatialTemporal.STAN.STAN
    :members:


STGCN
----------

.. autoclass:: epilearn.models.SpatialTemporal.STGCN.STGCN
    :members:


Code Example
-------------

.. code-block:: python

    import torch
    from epilearn.models.SpatialTemporal import ColaGNN

    num_nodes=47
    num_features = 1
    lookback = 16 # inputs size
    horizon = 3 # predicts size


    graph = torch.round(torch.rand((num_nodes, num_nodes)))
    features = torch.round(torch.rand((10, lookback, num_nodes, num_features)))
    node_target = torch.round(torch.rand((10, horizon, num_nodes))) 

    model=ColaGNN(num_nodes = num_nodes, num_features=num_features, num_timesteps_input=lookback, num_timesteps_output=horizon, device='cpu')
    model.fit(
            train_input=features, 
            train_target=node_target, 
            train_graph=graph,
            val_input=None,
            val_target=None,
            val_graph=None,
            epochs=20,
            loss='mse'
            )

