import xgboost as xgb
from sklearn.metrics import mean_squared_error
import torch.nn.init as init
import torch

class XGBModel:
    """
        Extreme Gradient Boosting (XGBoost) Model
        
        Parameters
        ----------
        num_features : int
            Number of features in each timestep of the input data.
        num_timesteps_input : int
            Number of timesteps considered for each input sample.
        num_timesteps_output : int
            Number of output timesteps to predict.
        n_estimators : int, optional
            Number of gradient boosted trees. Equivalent to the number of boosting rounds. Default: 40.
        learning_rate : float, optional
            Step size shrinkage used in update to prevents overfitting. Default: 0.1.
        max_depth : int, optional
            Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. Default: 5.
        reg_lambda : float, optional
            L2 regularization term on weights. Increasing this value will make model more conservative. Default: 1.0.
        reg_alpha : float, optional
            L1 regularization term on weights. Increasing this value will make model more conservative. Default: 0.1.

        Returns
        -------
        torch.Tensor
            A tensor of shape (batch_size, num_timesteps_output) representing the predicted values for the future timesteps.
            Each element corresponds to a predicted value for a future timestep.
        
    """
    def __init__(self, num_features, num_timesteps_input, num_timesteps_output,
                 n_estimators=40, learning_rate=0.1, max_depth=5, reg_lambda=1.0, reg_alpha=0.1):
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.model = None

    def get_features(self, X):
        features = X.reshape(-1, self.num_features * self.num_timesteps_input)
        return features

    def fit(self, train_input, train_target, val_input=None, val_target=None, epochs=1000, batch_size=10, verbose=False, patience=100):
        train_input = self.get_features(train_input)
        train_target = train_target.numpy() if isinstance(train_target, torch.Tensor) else train_target

        if val_input is not None and val_target is not None:
            val_input = self.get_features(val_input)
            val_target = val_target.numpy() if isinstance(val_target, torch.Tensor) else val_target

        self.model = xgb.XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=self.n_estimators, 
            learning_rate=self.learning_rate, 
            max_depth=self.max_depth,
            reg_lambda=self.reg_lambda,  
            reg_alpha=self.reg_alpha    
        )

        if val_input is not None and val_target is not None:
            self.model.fit(train_input, train_target, eval_set=[(train_input, train_target), (val_input, val_target)], verbose=verbose)
        else:
            self.model.fit(train_input, train_target)

        # Optionally, print MSE if verbose
        if verbose:
            train_pred = self.model.predict(train_input)
            train_mse = mean_squared_error(train_target, train_pred)
            print("Training MSE: ", train_mse)
            if val_input is not None and val_target is not None:
                val_pred = self.model.predict(val_input)
                val_mse = mean_squared_error(val_target, val_pred)
                print("Validation MSE: ", val_mse)

    def predict(self, feature):
        feature = self.get_features(feature)
        prediction = self.model.predict(feature)

        return torch.tensor(prediction)
