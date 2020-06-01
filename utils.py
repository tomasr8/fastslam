import math
import numpy as np
from typing import List
from particle import Particle


def dist(a: list, b: list) -> float:
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def neff(particles: List[Particle]) -> float:
    weights = [p.w for p in particles]
    return 1. / np.sum(np.square(weights))
