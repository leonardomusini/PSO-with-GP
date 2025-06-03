import numpy as np


# Safe functions for arithmetic operations in GP
def to_array(x, shape=None):
    x = np.asarray(x, dtype=np.float32)
    if shape:
        x = np.broadcast_to(x, shape)
    return x

def safe_binary_op(x, y, op, shape=None):
    try:
        x = to_array(x, shape)
        y = to_array(y, shape)
        result = op(x, y)
        result = np.asarray(result, dtype=np.float32)
        return result
    except Exception:
        fallback = np.ones(shape, dtype=np.float32) if shape else np.array([1.0], dtype=np.float32)
        return fallback

def valid_add(x, y, shape=None):
    return safe_binary_op(x, y, np.add, shape)

def valid_sub(x, y, shape=None):
    return safe_binary_op(x, y, np.subtract, shape)

def valid_mul(x, y, shape=None):
    return safe_binary_op(x, y, np.multiply, shape)

def valid_div(x, y, shape=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = safe_binary_op(x, y, np.divide, shape)
        return np.where(np.isfinite(result), result, 1.0)