import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    m = X.shape[0] # number of training samples
    n = X.shape[1] # number of parameters
    w = np.zeros(n)
    b = 0.0
    for i in range(steps): 
        z = w @ X.T + b
        p = _sigmoid(z)
        dj_dw = (X.T @ (p - y)) / m
        w = w - lr * dj_dw
        dj_db =  np.sum(p - y) / m
        b = b - lr * dj_db
    return w, b