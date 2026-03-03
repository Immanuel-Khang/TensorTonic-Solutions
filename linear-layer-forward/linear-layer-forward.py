import numpy as np
import numpy as np

def linear_layer_forward(x, w, b):
    """
    Manual forward pass using full row and full column.
    Must return a Python list.
    """
    
    X = np.array(x)
    W = np.array(w)
    b = np.array(b)
    
    n = X.shape[0]      # number of samples
    k = W.shape[1]      # number of output neurons
    
    res = np.zeros((n, k))
    
    for r in range(n):
        for c in range(k):
            # full row dot full column
            res[r, c] = X[r, :] @ W[:, c] + b[c]
    
    return res.tolist()