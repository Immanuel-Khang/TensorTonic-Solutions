import numpy as np
def polynomial_features(values, degree):
    """
    Generate polynomial features for each value up to the given degree.
    """
    result = []
    for val in values: 
        temp = []
        for deg in range(degree + 1): 
            temp.append(val**deg)
        result.append(temp)

    return result
    # Write code here