import numpy as np
def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    # Write code here
    min = np.min(data, axis=0, keepdims=True)
    max = np.max(data, axis=0, keepdims=True)
    range = max - min
    range[range == 0] = 1
    data = (data - min) / (range)

    return data.tolist()