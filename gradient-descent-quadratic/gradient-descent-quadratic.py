def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    x = x0 # starting point
    for i in range(steps): 
        df_dx = 2 * a * x + b
        x = x - lr * df_dx

    return x
    pass