### Functions in this file are for debugging purpose
### Linjie Yang

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # defalut: last dimension of x is the score dimension
    axis = len(x.shape) - 1
    x = x - x.max(axis = axis, keepdims=True)
    sf = np.exp(x)
    sf = sf / np.sum(sf, axis=axis, keepdims=True)
    return sf