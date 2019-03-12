import numpy as np


def trim(input_array, trim=-1):
    """Trim padding values at tail."""
    return input_array[~np.in1d(input_array, trim).reshape(input_array.shape)]


def rescale(input_array, scale=255):
    return input_array/scale


