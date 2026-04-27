import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    # Write code here
    x = np.array(x)
    h_prev = np.array(h_prev)

    x, reshapeX = _as2d(x, x.shape[-1])
    h_prev, reshapeH = _as2d(h_prev, h_prev.shape[-1])

    Wz = np.array(params["Wz"])
    Wr = np.array(params["Wr"])
    Wh = np.array(params["Wh"])

    Uz = np.array(params["Uz"])
    Ur = np.array(params["Ur"])
    Uh = np.array(params["Uh"])

    bz = np.array(params["bz"])
    br = np.array(params["br"])
    bh = np.array(params["bh"])

    bz, zReshape = _as2d(bz, bz.shape[-1])
    br, rReshape = _as2d(br, br.shape[-1])
    bh, hReshape = _as2d(bh, bh.shape[-1])

    
    z_t = _sigmoid(x @ Wz + h_prev @ Uz + bz)

    r_t = _sigmoid(x @ Wr + h_prev @ Ur + br)

    h_tilde = np.tanh(x @ Wh + (r_t * h_prev) @ Uh + bh)

    h_next = (1 - z_t) * h_prev + z_t * h_tilde

    if reshapeH == True: return h_next.flatten()
    return h_next
    