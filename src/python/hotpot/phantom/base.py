import numpy as np
import tables
from skimage.measure import block_reduce
from tables import IsDescription, Float32Col

def cos(deg):
    return np.cos(-1 * deg * (np.pi / 180))


def sin(deg):
    return np.sin(deg * (np.pi / 180))


def tan(deg):
    return np.tan(deg * (np.pi / 180))


def rotate(p, th):
    """
    usage: rotate(P, 60)
    """
    p0 = (P_SIZE / 2, P_SIZE / 2)

    return ((p[0] - p0[0]) * cos(th) - (p[1] - p0[1]) * sin(th) + p0[0],
            (p[0] - p0[0]) * sin(th) + (p[1] - p0[1]) * cos(th) + p0[1])