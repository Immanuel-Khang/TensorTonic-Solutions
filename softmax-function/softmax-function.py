import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.array(x)
    
    if x.ndim == 1:
        z = np.exp(x - np.max(x))
        return z / np.sum(z)

    z = np.exp(x - np.max(x, axis = 1, keepdims = True))
    res =  z / np.sum(z, axis = 1, keepdims = True)
    return res
    pass