import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = y_true.shape[0]
    
    correct_probs = y_pred[np.arange(n), y_true]
    res = -np.mean(np.log(correct_probs))
    return res
    pass