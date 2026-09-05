import numpy as np
import torch
import warnings
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


class BaseScikitModel:
    """
    Base class for scikit-learn models adapted for time series forecasting.
    Provides a unified interface compatible with the epilearn framework.
    """
    def __init__(self, num_features, num_timesteps_input, num_timesteps_output,
                 device='cpu', **model_params):
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.device = device
        self._nowcast = bool(model_params.pop('nowcast', False))
        self.model_params = model_params
        self.models = []  # One model per output timestep

    def _prepare_data(self, data, is_tensor=True):
        """Convert tensor to numpy if needed."""
        if is_tensor and hasattr(data, 'numpy'):
            # Move to CPU if on CUDA before converting to numpy
            if hasattr(data, 'is_cuda') and data.is_cuda:
                return data.cpu().numpy()
            return data.numpy()
        return data

    def _create_model(self):
        """Create a new instance of the sklearn model. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_model")

    def _reshape_for_sklearn(self, input_data):
        """
        Reshape sequential data for sklearn models.
        Input: (batch_size, num_timesteps, num_features)
        Output: (batch_size, num_timesteps * num_features)
        """
        batch_size = input_data.shape[0]
        return input_data.reshape(batch_size, -1)

    def fit(self, train_input, train_target, train_states=None, val_input=None, val_target=None,
            train_graph=None, train_dynamic_graph=None, val_graph=None, val_dynamic_graph=None, 
            val_states=None, epochs=1000, batch_size=10, verbose=False, patience=100, 
            lr=None, weight_decay=None, loss=None, initialize=True, **kwargs):
        """
        Fit the model to training data.
        
        Parameters
        ----------
        train_input : torch.Tensor
            Shape (batch_size, num_timesteps_input, num_features)
        train_target : torch.Tensor
            Shape (batch_size, num_timesteps_output)
        verbose : bool
            If True, prints progress and error messages
        **kwargs : dict
            Additional parameters (ignored for compatibility)
        
        Returns
        -------
        list
            Forecasted values for the training data
        """
        # Convert to numpy
        train_input = self._prepare_data(train_input)
        train_target = self._prepare_data(train_target)
        
        # Handle different target shapes
        # Could be: (batch_size, num_timesteps) or (batch_size, num_nodes, num_timesteps)
        if len(train_target.shape) == 1:
            train_target = train_target.reshape(-1, 1)
        elif len(train_target.shape) == 3:
            # Shape: (batch_size, num_nodes, num_timesteps) -> (batch_size, num_timesteps)
            # We take the first node for single-node forecasting
            train_target = train_target[:, 0, :]
        
        # Now train_target is (batch_size, num_timesteps_output)
        num_outputs = train_target.shape[1] if len(train_target.shape) == 2 else 1
        
        # Reshape input for sklearn
        X_train = self._reshape_for_sklearn(train_input)
        
        # Train one model per output timestep for multi-step forecasting
        self.models = []
        for t in range(num_outputs):
            model = self._create_model()
            y_train = train_target[:, t]
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train)
                self.models.append(model)
            except Exception as e:
                if verbose:
                    print(f"Training failed for timestep {t}: {str(e)}")
                self.models.append(None)
        
        # Generate predictions
        predictions = self.predict(torch.tensor(train_input, dtype=torch.float32))
        
        # Compute MSE
        if verbose:
            # predictions shape: (batch_size, num_timesteps)
            # train_target shape: (batch_size, num_timesteps)
            mse_loss = np.mean((train_target - predictions.numpy()) ** 2)
            print(f"MSE Loss: {mse_loss:.6f}")
        
        # Convert to list maintaining the shape
        # predictions is (batch_size, num_timesteps)
        return predictions.numpy().tolist()

    def predict(self, feature, graph=None, states=None, dynamic_graph=None, **kwargs):
        """
        Make predictions on new data.
        
        Parameters
        ----------
        feature : torch.Tensor
            Shape (batch_size, num_timesteps_input, num_features) or
            (batch_size, num_nodes, num_timesteps_input, num_features)
        **kwargs : dict
            Additional parameters (ignored for compatibility)
        
        Returns
        -------
        torch.Tensor
            Shape matching the target: (batch_size, num_timesteps_output) or 
            (batch_size, num_nodes, num_timesteps_output)
        """
        test_input = self._prepare_data(feature)
        
        # Check if input has node dimension
        has_node_dim = (len(test_input.shape) == 4)

        # Handle node dimension: take first node for single-node forecasting.
        # NOTE: In the benchmark pipeline, 4D data is reshaped to 3D upstream
        # by _reshape_4d_to_temporal(), so this branch should not be reached.
        # If it is, it silently drops all nodes except the first.
        if has_node_dim:
            import warnings
            warnings.warn(
                f"ScikitModel.predict() received 4D input {test_input.shape}; "
                f"only node 0 will be used. This usually means the pipeline "
                f"did not reshape the data upstream.",
                stacklevel=2,
            )
            test_input = test_input[:, 0, :, :]
        
        X_test = self._reshape_for_sklearn(test_input)
        
        # Predict for each timestep using trained models
        predictions = []
        
        for model in self.models:
            if model is not None:
                try:
                    pred = model.predict(X_test)
                    # Check for NaN and handle
                    if np.any(np.isnan(pred)):
                        pred = np.nan_to_num(pred, nan=0.0)
                    predictions.append(pred)
                except Exception as e:
                    predictions.append(np.zeros(X_test.shape[0]))
            else:
                predictions.append(np.zeros(X_test.shape[0]))
        
        # Stack predictions: (num_timesteps, batch_size) -> (batch_size, num_timesteps)
        predictions_array = np.array(predictions).T
        result = torch.tensor(predictions_array, dtype=torch.float32)
        
        # Only add node dimension if input had node dimension
        # For temporal models, we want (batch_size, num_timesteps)
        # For spatiotemporal models with nodes, we want (batch_size, 1, num_timesteps)
        if has_node_dim:
            result = result.unsqueeze(1)  # (batch_size, num_timesteps) -> (batch_size, 1, num_timesteps)
        
        # Ensure contiguous memory layout for .view() operations
        return result.contiguous()

    def to(self, device):
        """Compatibility method for device placement."""
        self.device = device
        return self


class LinearRegressionModel(BaseScikitModel):
    """
    Linear Regression Model for time series forecasting.
    Wrapper around sklearn.linear_model.LinearRegression.
    
    Parameters
    ----------
    num_features : int
        Number of features in each timestep
    num_timesteps_input : int
        Number of input timesteps (lookback window)
    num_timesteps_output : int
        Number of output timesteps to predict (forecast horizon)
    fit_intercept : bool, optional
        Whether to calculate the intercept (default: True)
    """
    
    def _create_model(self):
        """Create a LinearRegression model."""
        fit_intercept = self.model_params.get('fit_intercept', True)
        return LinearRegression(fit_intercept=fit_intercept)


class RidgeModel(BaseScikitModel):
    """
    Ridge Regression Model with L2 regularization.
    Wrapper around sklearn.linear_model.Ridge.
    
    Parameters
    ----------
    alpha : float, optional
        Regularization strength (default: 1.0)
    """
    
    def _create_model(self):
        """Create a Ridge model."""
        alpha = self.model_params.get('alpha', 1.0)
        return Ridge(alpha=alpha)


class LassoModel(BaseScikitModel):
    """
    Lasso Regression Model with L1 regularization.
    Wrapper around sklearn.linear_model.Lasso.
    
    Parameters
    ----------
    alpha : float, optional
        Regularization strength (default: 1.0)
    """
    
    def _create_model(self):
        """Create a Lasso model."""
        alpha = self.model_params.get('alpha', 1.0)
        return Lasso(alpha=alpha, max_iter=10000)


class ElasticNetModel(BaseScikitModel):
    """
    ElasticNet Regression Model with L1 and L2 regularization.
    Wrapper around sklearn.linear_model.ElasticNet.
    
    Parameters
    ----------
    alpha : float, optional
        Regularization strength (default: 1.0)
    l1_ratio : float, optional
        Mix ratio between L1 and L2 (default: 0.5)
    """
    
    def _create_model(self):
        """Create an ElasticNet model."""
        alpha = self.model_params.get('alpha', 1.0)
        l1_ratio = self.model_params.get('l1_ratio', 0.5)
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)


class RandomForestModel(BaseScikitModel):
    """
    Random Forest Regressor for time series forecasting.
    Wrapper around sklearn.ensemble.RandomForestRegressor.
    
    Parameters
    ----------
    n_estimators : int, optional
        Number of trees (default: 100)
    max_depth : int, optional
        Maximum depth of trees (default: None)
    min_samples_split : int, optional
        Minimum samples required to split (default: 2)
    """
    
    def _create_model(self):
        """Create a RandomForest model."""
        n_estimators = self.model_params.get('n_estimators', 100)
        max_depth = self.model_params.get('max_depth', None)
        min_samples_split = self.model_params.get('min_samples_split', 2)
        random_state = self.model_params.get('random_state', 42)
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )


class GradientBoostingModel(BaseScikitModel):
    """
    Gradient Boosting Regressor for time series forecasting.
    Wrapper around sklearn.ensemble.GradientBoostingRegressor.
    
    Parameters
    ----------
    n_estimators : int, optional
        Number of boosting stages (default: 100)
    learning_rate : float, optional
        Learning rate (default: 0.1)
    max_depth : int, optional
        Maximum depth of trees (default: 3)
    """
    
    def _create_model(self):
        """Create a GradientBoosting model."""
        n_estimators = self.model_params.get('n_estimators', 100)
        learning_rate = self.model_params.get('learning_rate', 0.1)
        max_depth = self.model_params.get('max_depth', 3)
        random_state = self.model_params.get('random_state', 42)
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )


class SVRModel(BaseScikitModel):
    """
    Support Vector Regressor for time series forecasting.
    Wrapper around sklearn.svm.SVR.
    
    Parameters
    ----------
    kernel : str, optional
        Kernel type ('linear', 'poly', 'rbf', 'sigmoid') (default: 'rbf')
    C : float, optional
        Regularization parameter (default: 1.0)
    epsilon : float, optional
        Epsilon in epsilon-SVR (default: 0.1)
    """
    
    def _create_model(self):
        """Create an SVR model."""
        kernel = self.model_params.get('kernel', 'rbf')
        C = self.model_params.get('C', 1.0)
        epsilon = self.model_params.get('epsilon', 0.1)
        return SVR(kernel=kernel, C=C, epsilon=epsilon)


class KNNModel(BaseScikitModel):
    """
    K-Nearest Neighbors Regressor for time series forecasting.
    Wrapper around sklearn.neighbors.KNeighborsRegressor.
    
    Parameters
    ----------
    n_neighbors : int, optional
        Number of neighbors (default: 5)
    weights : str, optional
        Weight function ('uniform', 'distance') (default: 'uniform')
    """
    
    def _create_model(self):
        """Create a KNN model."""
        n_neighbors = self.model_params.get('n_neighbors', 5)
        weights = self.model_params.get('weights', 'uniform')
        return KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, n_jobs=-1)


class DecisionTreeModel(BaseScikitModel):
    """
    Decision Tree Regressor for time series forecasting.
    Wrapper around sklearn.tree.DecisionTreeRegressor.
    
    Parameters
    ----------
    max_depth : int, optional
        Maximum depth of tree (default: None)
    min_samples_split : int, optional
        Minimum samples required to split (default: 2)
    """
    
    def _create_model(self):
        """Create a DecisionTree model."""
        max_depth = self.model_params.get('max_depth', None)
        min_samples_split = self.model_params.get('min_samples_split', 2)
        random_state = self.model_params.get('random_state', 42)
        return DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
