import numpy as np

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / (e_x.sum(axis=0)+ 0.0000000000000000001)  # Normalize

def relu(x):
    """Rectified Linear Unit (ReLU) activation function."""
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(x.dtype)

def softmax_derivative(s):
    """
    Derivative of softmax for each vector independently.
    s must be the softmax output.
    Returns the Jacobian for each vector.
    - If s is shape (n,): returns (n, n)
    - If s is shape (batch, n): returns (batch, n, n)
    """
    if s.ndim == 1:
        # single vector
        return np.diag(s) - np.outer(s, s)
    else:
        # batch of vectors
        batch = s.shape[0]
        n = s.shape[1]
        J = np.zeros((batch, n, n))
        for i in range(batch):
            si = s[i]
            J[i] = np.diag(si) - np.outer(si, si)
        return J