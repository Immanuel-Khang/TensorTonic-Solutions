import numpy as np

def manhattan_distance(x: list, y: list) -> float:
    """
    Return the Manhattan distance between x and y.
    """
    # Write code here
    x = np.asarray(x)
    y = np.asarray(y)

    z = abs(x - y)
    return float(np.sum(z))
    pass