import numpy as np

def xavier_init(size):
    """Xavier initialization for neural network weights."""
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size=size, scale=xavier_stddev).astype(np.float32)