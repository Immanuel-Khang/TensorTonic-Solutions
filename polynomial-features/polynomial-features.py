import numpy as np
def polynomial_features(values, degree):
    """
    Generate polynomial features for each value up to the given degree.
    """
    values = np.asarray(values)
    return (values[:, None] ** np.arange(degree + 1)).tolist()

    
    # Write code here