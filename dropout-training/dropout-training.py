import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p (drop probability).
    Return (output, dropout_pattern).
    """
    x = np.array(x)

    if rng is None:
        rng = np.random.default_rng()

    keep_prob = 1 - p

    # Scaled (inverted dropout) mask
    mask = (rng.random(x.shape) < keep_prob) / keep_prob

    out = x * mask

    return out, mask