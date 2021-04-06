import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


odom = np.load("odom.npy")
odom[:, 2] += np.pi/2
odom[:, 2] = np.arctan2(np.sin(odom[:, 2]), np.cos(odom[:, 2]))

control = np.zeros((len(odom), 3), dtype=np.float)
control[0] = [0, 0, odom[0, 3]]

for i in range(1, len(odom)):
    start = odom[i-1]
    end = odom[i]
    
    stamp = end[-1]
    angle = np.arctan2(np.sin(end[2] - start[2]), np.cos(end[2] - start[2]))
    angle += np.random.normal(0, 0.01)
    angle = wrap_angle(angle)

    dist = np.linalg.norm(start[:2] - end[:2]) + np.random.normal(0, 0.1)

    control[i] = [angle, dist, stamp]


np.save("control.npy", control)