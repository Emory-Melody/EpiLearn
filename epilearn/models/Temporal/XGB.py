import xgboost as xgb
from sklearn.metrics import mean_squared_error
import torch.nn.init as init
import torch

class XGBModel:
    def __init__(self, num_features, num_timesteps_input, num_timesteps_output, nhid=256, dropout=0.5, use_norm=False):
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.nhid = nhid
        self.dropout = dropout
        self.use_norm = use_norm
        self.model = None  # Initialize the model variable here

    def get_features(self, X):
        # Assuming X is a NumPy array; reshape it according to num_features and num_timesteps_input
        return X.reshape(-1, self.num_features * self.num_timesteps_input)

    def fit(self, train_input, train_target, val_input=None, val_target=None, epochs=1000, batch_size=10, verbose=False, patience=100):
        train_input_np = train_input.numpy()
        train_target_np = train_target.numpy()
        train_input = self.get_features(train_input)
        
        if val_input is not None and val_target is not None:
            val_input_np = val_input.numpy()
            val_target_np = val_target.numpy()
            val_input = self.get_features(val_input)

        self.model = xgb.XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=40, 
            learning_rate=0.1, 
            max_depth=5,
            reg_lambda=1.0,  # L2 regularization factor
            reg_alpha=0.1    # L1 regularization factor
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
        test_input = self.get_features(feature)
        prediction = self.model.predict(test_input)

        return torch.tensor(prediction)


