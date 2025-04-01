from .SIR import *
from . import (
    ARIMA,
    Dlinear,
    EINN,
    GRU,
    LSTM,
    # XGB,
    CNN
)

from .ARIMA import VARMAXModel, ARIMAModel
from .Dlinear import DlinearModel
# from .EINN import EINN
from .GRU import GRUModel
from .LSTM import LSTMModel
# from .XGB import XGBModel
from .CNN import CNNModel