def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Write code here
    L = (6/(fan_in + fan_out))**0.5

    for i in range(len(W)):
        for j in range(len(W[i])):
            W[i][j] = W[i][j] * 2 * L - L

    return W