from __future__ import division
import numpy as np
from numpy import linalg

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=0.1):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))
    