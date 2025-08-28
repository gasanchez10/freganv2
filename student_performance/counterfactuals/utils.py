import tensorflow as tf
import numpy as np

def xavier_init(size):
    """Initialize neural network weights using Xavier initialization.
    
    Provides better weight initialization for deep networks by scaling
    the random values based on the input dimension.
    """
    in_dim = size[0]
    # Calculate standard deviation for Xavier initialization
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    
    return tf.random.normal(shape=size, stddev=xavier_stddev)