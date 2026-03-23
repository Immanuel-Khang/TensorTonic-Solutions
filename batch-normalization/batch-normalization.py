import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    x = np.array(x)
   
    gamma = np.array(gamma)
    beta = np.array(beta)

    if x.ndim == 2: 
        m = x.shape[0]
        u = (1 / m) * np.sum(x, axis=0, keepdims=True)
        var = (1 / m) * np.sum((x - u)**2, axis=0, keepdims=True)
        x_temp = (x - u) / np.sqrt(var + eps)
        y = gamma * x_temp + beta
        return y

    m = x.shape[0] * x.shape[2] * x.shape[3]
    u = (1 / m) * np.sum(x, axis=(0,2,3), keepdims=True)
    var = (1 / m) * np.sum((x - u)**2, axis=(0,2,3), keepdims=True)
    x_temp = (x - u) / np.sqrt(var + eps)

    gamma = gamma.reshape(1, -1, 1, 1)
    beta = beta.reshape(1, -1, 1, 1)

    y = gamma * x_temp + beta
    return y
    pass