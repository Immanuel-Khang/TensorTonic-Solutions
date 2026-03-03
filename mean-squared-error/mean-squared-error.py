import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    y_pred = np.array(y_pred)
    n = y_pred.shape[0]
    y_true = np.array(y_true)
    res = np.sum((y_pred - y_true) ** 2) / n
    return res
