import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.asarray(X)
    y = np.asarray(y)
    y = y.reshape(-1, 1)
    
    samples, features = X.shape
    w = np.array([0] * features)
    b = 0
    w = w.reshape(-1, 1)
    
    for step in range(steps):         
        z = X @ w + b
        u = _sigmoid(z)
        dL_dz = (1 / samples) * (u - y)

        dL_dz = dL_dz.reshape(-1, 1)
        
        dL_dw = X.T @ dL_dz
        dL_db = np.sum(dL_dz, axis=0)
        
        w = w - lr * dL_dw
        b = b - lr * dL_db

    b = b[0]
    return w.flatten(), b
    pass