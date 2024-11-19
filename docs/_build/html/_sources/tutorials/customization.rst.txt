Model&Dataset Customization
===================================

Dataset Customization
-----------------------

EpiLearn povides UniversalDataset class to load all datasets, including time series data, static graph, and dynamic graphs.

For temporal tasks, we need at least two inputs: time series features (**[Length, Channels]**) and prediction target (**[Length, 1]**). \ 
Note that the prediction target can be inlcuded as one of the channels of the time series features. If you only have univariate time series data, then the input x and y will be the **same**.

.. code-block:: python

    dataset = UniversalDataset(x=features, y=targets)

For spatial-temporal tasks, we also load static graph and dynamic graph. In this case, the shapes of inputs change to x: **[Length, num_nodes, channels]**; \
y: **[Length, num_nodes]**; graph: **[num_nodes, num_nodes]**; dynamic_graph: **[Length, num_nodes, num_nodes]**.

.. code-block:: python

    dataset = UniversalDataset(x=node_features, y=targets, graph=static_graph, dynamic_graph=dynamic_graph)

For spatial tasks, we are using a period of future states to predict a certain point of history states of the nodes. In this case, In this case, the shapes of inputs change to \ 
x: **[num_samples, num_nodes, channels]**; y: **[num_samples, num_nodes]**; graph: **[num_nodes, num_nodes]**; dynamic_graph: **[num_samples, num_nodes, num_nodes]**. Note that \ 
although the inputs shapes are the same as the spatial-temporal task, the meaning of the first dim is different, denoting number of samples instead of the length the time series. \ 
This means you can input multiple same(graph) or different(dynamic_graph) graph samples.

.. code-block:: python
    
    dataset = UniversalDataset(x=node_features, y=targets, graph=same_graph_each_sample, dynamic_graph=defferent_graph_each_sample)

For more coding details, please refer to `Dataset Customization <https://github.com/Emory-Melody/EpiLearn/blob/main/examples/dataset_customization.ipynb>`_.


Model Customization
-----------------------

EpiLearn builds base classes for temporal, spatial, and spatial-temporal models. To build your own customized models, the model must also inherit from the corresponding base model.
Here we provide an example of building a cusotmized LSTM model. For more information, please refer to `Model Customization <https://github.com/Emory-Melody/EpiLearn/blob/main/examples/model_customization.ipynb>`_.

**Step 1**
Import the base class corresponding to yout task. For temporal task, we import the tempora base class, which is the BaseModel shown below.

.. code-block:: python

    from epilearn.models.Temporal.base import BaseModel

**Step 2**
Build your new model class inherited from the BaseModel. Then, define your __init__ function to pass the hyperparameters used to initialize your model. \
Note that the names of the hyperparameters are not fixed, the default names below will be used if you do not pass your hyper parameters in **Step 5**.

.. code-block:: python

    class CustomizedTemporal(BaseModel):
            def __init__(self,
                        num_features,
                        num_timesteps_input,
                        num_timesteps_output,
                        hidden_size,
                        num_layers,
                        bidirectional,
                        device = 'cpu'):
                super(CustomizedTemporal, self).__init__(device=device)
                self.num_feats = num_features
                self.hidden = hidden_size
                self.num_layers = num_layers
                self.bidirectional=bidirectional
                self.lookback = num_timesteps_input
                self.horizon = num_timesteps_output
                self.device = device

                self.lstm = nn.LSTM(input_size=self.num_feats, hidden_size=self.hidden, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional)
                self.fc = nn.Linear(self.hidden, self.horizon)

**Step 3**

Next, define the forward function of your model. This function defines how the input data is processed. As long as your dataset is initilized correctly, the input of this function is fixed. \
The output of this function is not fixed, but we require a fixed format. For example, for regression tasks, the output shape will be the same as the label, which is [batch, horizon, 1] (temporal) or [batch, horizon, num_nodes] (spatial-temporal). \
Below is an example of the forward function of the model in **Step 2**.

.. code-block:: python

        def forward(self, feature, graph=None, states=None, dynamic_graph=None, **kargs):        
            # Forward propagate LSTM
            out, _ = self.lstm(feature)  # out: tensor of shape (batch, seq_length, hidden_size * num_directions)
            
            # Decode the last hidden state
            out = self.fc(out[:, -1, :])

            return out

**Step 4 (Optional)**
The initialization method of your model can be define here.

.. code-block:: python

        def initialize(self):
            for name, param in self.lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)


**Step 5**
After your model class is defined, we can apply the same pipeline in `Quickstart <https://epilearn-doc.readthedocs.io/en/latest/Quickstart.html>`_

.. code-block:: python

    # Generate cosine data
    t = torch.linspace(0, 1, 500)
    cos_wave = 1 * torch.cos(2 * torch.pi * 3 * t)

    # Initialize dataset with generated data
    inputs = cos_wave.reshape(-1, 1)
    dataset = UniversalDataset(x=inputs, y=inputs)

    # Initialize forecast settings
    lookback = 36 # inputs size
    horizon = 3 # predicts size

    # Adding Transformations
    transformation = transforms.Compose({
                    "features": [transforms.normalize_feat()],
                    "target": [transforms.normalize_feat()]
                    })
    dataset.transforms = transformation

    # Initialize Task
    task = Forecast(prototype=CustomizedTemporal,
                    dataset=None,
                    lookback=lookback,
                    horizon=horizon,
                    device='cpu')


    # Define hyperparameters of your model
    model_args = {"num_features": 1, "num_timesteps_input": lookback, "num_timesteps_output": horizon, "hidden_size": 16, "num_layers": 2, "bidirectional": False, "device": 'cpu'}

    # Training
    result = task.train_model(dataset=dataset,
                            loss='mse',
                            epochs=40,
                            batch_size=8,
                            train_rate=0.6,
                            val_rate=0.1,
                            lr=1e-3,
                            permute_dataset=False,
                            model_args=model_args) # pass the hyperparameters of your model





