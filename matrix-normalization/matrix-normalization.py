import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    matrix = np.array(matrix, dtype=float)

    if matrix.ndim != 2:
        return None

    if axis is not None and axis >= matrix.ndim:
            return None

    if norm_type == 'l2':
        divisor = np.sqrt(np.sum(matrix**2, axis=axis, keepdims=True))

    elif norm_type == 'l1':
        divisor = np.sum(np.abs(matrix), axis=axis, keepdims=True)

    elif norm_type == 'max':
        divisor = np.max(np.abs(matrix), axis=axis, keepdims=True)
    else : 
        return None
    divisor[divisor == 0] = 1
    return matrix / divisor