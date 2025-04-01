import torch

from ..utils import utils, metrics  
from .base import BaseTask

class Detection(BaseTask):
    """
    The Detection class is designed to handle the training and evaluation of models for detection tasks.
    It extends the BaseTask class, incorporating specific functionalities to work with spatial-temporal
    data models. This class includes methods for model training, evaluation, and data preprocessing to
    facilitate experiments with different types of neural network architectures and configurations.
    """
    def __init__(self, prototype = None, model = None, dataset = None, lookback = None, horizon = None, device = 'cpu'):
        super().__init__(prototype, model, dataset, lookback, horizon, device)

    def train_model(self,
                    dataset=None,
                    config=None,
                    permute_dataset=True,
                    train_rate=0.6,
                    val_rate=0.2,
                    loss='ce', 
                    epochs=1000, 
                    batch_size=10,
                    lr=1e-3, 
                    region_idx=None,
                    initialize=True, 
                    verbose=False, 
                    patience=100, 
                    model_args={},
                    ):
        '''
        Trains the detection model using the provided dataset and configuration settings. It handles data splitting, model initialization, and the training process, and also evaluates the model on the test set, reporting the accuracy.
        '''
        if config is not None:
            permute_dataset = config.permute
            train_rate = config.train_rate
            val_rate = config.val_rate
            loss = config.loss
            epochs=config.epochs
            batch_size=config.batch_size
            lr=config.lr
            initialize=config.initialize
            patience=config.patience

        if dataset is None:
            try:
                dataset = self.dataset
            except:
                raise RuntimeError("dataset not exists, please input dataset or use load_dataset() first!")
        else:
            self.dataset = dataset
        
        if not hasattr(self, "model"):
            raise RuntimeError("model not exists, please use load_model() to load model first!")

        self.train_split, self.val_split, self.test_split, self.adj = self.get_splits(self.dataset, train_rate, val_rate, region_idx, permute_dataset)

        try:
            self.target_mean, self.target_std = self.dataset.transforms.target_mean, dataset.transforms.target_std
        except:
            self.target_mean, self.target_std = 0, 1


        if len(model_args) != 0:
            self.model = self.prototype(**model_args)
            print("Model Initialized!")
        else:
            try:
                self.model = self.prototype(
                                            num_nodes=self.adj.shape[0],
                                            num_features=self.train_split['features'].shape[-1],
                                            num_timesteps_input=self.lookback,
                                            num_timesteps_output=self.horizon,
                                            device=self.device
                                            ).to(device=self.device)
                print("spatial-temporal model loaded!")
            except:
                try:
                    self.model = self.prototype( 
                                            num_features=self.train_split['features'].shape[2],
                                            num_timesteps_input=self.lookback,
                                            num_timesteps_output=self.horizon,
                                            device=self.device).to(device=self.device)
                                            
                    print("temporal model loaded!")
                except:
                    if len(self.train_split['features'].shape) == 4:
                        num_features = self.train_split['features'].shape[-1] * self.train_split['features'].shape[-2]
                    else:
                        num_features = self.train_split['features'].shape[-1]
                    self.model = self.prototype(
                                            num_features=num_features,
                                            num_classes=self.horizon,
                                            device=self.device).to(device=self.device)
                    print("spatial model loaded!")
        

        # train
        self.model.fit(
                train_input=self.train_split['features'], 
                train_target=self.train_split['targets'], 
                train_states = self.train_split['states'],
                train_graph=self.adj, 
                train_dynamic_graph=self.train_split['dynamic_graph'],
                val_input=self.val_split['features'], 
                val_target=self.val_split['targets'], 
                val_states=self.val_split['states'],
                val_graph=self.adj,
                val_dynamic_graph=self.val_split['dynamic_graph'],
                verbose=verbose,
                lr=lr,
                batch_size=batch_size,
                epochs=epochs,
                loss=loss,
                initialize=initialize,
                patience=patience
                )
        

        # evaluate
        self.test_graph = self.adj
        self.test_feature = self.test_split['features']
        self.test_target = self.test_split['targets']
        self.test_states = self.test_split['states']
        self.test_dynamic_graph = self.test_split['dynamic_graph']
        # evaluate
        out = self.model.predict(feature=self.test_feature, graph=self.test_graph, states=self.test_states, dynamic_graph=self.test_dynamic_graph)
        if type(out) is tuple:
            out = out[0]
        preds = out.detach().cpu().argmax(2)
        # metrics
        acc = metrics.get_ACC(preds, self.test_target)
        print(f"Test ACC: {acc.item()}")
        
        return {'acc': acc, "predictions": preds, "targets": self.test_target}


    def evaluate_model(self,
                    model=None,
                    features=None,
                    graph=None,
                    dynamic_graph=None,
                    states=None,
                    targets=None,
                    ):
        '''
        Evaluates the trained model on a new dataset or using preloaded features and graphs. It outputs the prediction accuracy of the model.
        '''
        if model is None:
            if not hasattr(self, "model"):
                raise RuntimeError("model not exists, please use load_model() to load model first!")
            model = self.model
        
        features = self.test_feature if features is None else features
        graph = self.test_graph if graph is None else graph
        states = self.test_states if states is None else states
        dynamic_graph = self.test_dynamic_graph if dynamic_graph is None else dynamic_graph
        targets = self.test_target if targets is None else targets

        # evaluate
        out = self.model.predict(feature=features, graph=graph, states=states, dynamic_graph=dynamic_graph)
        preds = out.detach().cpu().argmax(-1)
        # metrics
        acc = metrics.get_ACC(preds, targets)
        print(f"ACC: {acc.item()}")
        return {'acc': acc, "predictions": preds, "targets": self.test_target}

    def get_splits(self, dataset=None, train_rate=0.6, val_rate=0.2, region_idx=None, permute=False):
        """
        Splits the provided dataset into training, validation, and testing sets based on specified rates. It also handles 
        preprocessing to normalize the data and prepare it for the model.
        """
        if dataset is None:
            try:
                dataset = self.dataset
            except:
                raise RuntimeError("dataset not exists, please use load_dataset() to load dataset first!")
            
        # preprocessing
        self.train_dataset, self.val_dataset, self.test_dataset = dataset.ganerate_splits(train_rate=train_rate, val_rate=val_rate)
        adj = dataset.graph

        return  {'features': self.train_dataset['features'], 'targets': self.train_dataset['target'], 'states': self.train_dataset['states'], 'dynamic_graph': self.train_dataset['dynamic_graph']}, \
                {'features': self.val_dataset['features'], 'targets': self.val_dataset['target'], 'states': self.val_dataset['states'], 'dynamic_graph': self.val_dataset['dynamic_graph']}, \
                {'features': self.test_dataset['features'], 'targets': self.test_dataset['target'], 'states': self.test_dataset['states'], 'dynamic_graph': self.test_dataset['dynamic_graph']}, \
                adj

        

