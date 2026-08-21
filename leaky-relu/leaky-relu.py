import numpy as np

def leaky_relu(x: list | float, alpha: float = 0.01) -> np.ndarray:
    """
    Apply Leaky ReLU elementwise and return a NumPy array.
    """
    # Write code here
    x = np.asarray(x)
    return np.where(x >= 0, x, alpha * x)
    pass