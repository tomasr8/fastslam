import math
import numpy as np


def dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def neff(particles):
    weights = [p.w for p in particles]
    return 1. / np.sum(np.square(weights))
