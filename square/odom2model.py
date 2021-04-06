import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

np.random.seed(0)

odom = np.vstack((
    np.array([
        np.linspace(1, 19, num=100),
        np.ones(shape=100),
        np.zeros(shape=100)
    ]).T,
    np.array([
        19 * np.ones(shape=10),
        np.ones(shape=10),
        np.linspace(0, np.pi/2, num=10)
    ]).T,

    np.array([
        19 * np.ones(shape=100),
        np.linspace(1, 19, num=100),
        np.pi/2 * np.ones(shape=100)
    ]).T,
    np.array([
        19 * np.ones(shape=10),
        19 * np.ones(shape=10),
        np.linspace(np.pi/2, np.pi, num=10)
    ]).T,

    np.array([
        np.flip(np.linspace(1, 19, num=100)),
        19 * np.ones(shape=100),
        np.pi * np.ones(shape=100)
    ]).T,
    np.array([
        np.ones(shape=10),
        19 * np.ones(shape=10),
        np.linspace(np.pi, 1.5*np.pi, num=10)
    ]).T,

    np.array([
        np.ones(shape=100),
        np.flip(np.linspace(1, 19, num=100)),
        1.5*np.pi * np.ones(shape=100)
    ]).T,
    np.array([
        np.ones(shape=10),
        np.ones(shape=10),
        np.linspace(1.5*np.pi, 2*np.pi, num=10)
    ]).T
))

for i in range(len(odom)):
    odom[i, 2] = wrap_angle(odom[i, 2])

odom = np.vstack((
    odom.copy(),
    odom.copy()
))


control = np.zeros((len(odom), 2), dtype=np.float)
control[0] = [0, 0]

Q = [0.001, 0.1]

for i in range(1, len(odom)):
    start = odom[i-1]
    end = odom[i]
    
    angle = wrap_angle(end[2] - start[2])
    angle += np.random.normal(0, Q[0])
    angle = wrap_angle(angle)

    dist = np.linalg.norm(start[:2] - end[:2])
    
    if dist > 0.1:
        dist += np.random.normal(0, Q[1])

    control[i] = [angle, dist]


dead_reckoning = [odom[0]]
for i in range(1, len(odom)):
    angle = wrap_angle(dead_reckoning[-1][2] + control[i, 0])

    x = dead_reckoning[-1][0] + np.cos(angle) * control[i, 1]
    y = dead_reckoning[-1][1] + np.sin(angle) * control[i, 1]

    dead_reckoning.append([ x, y, angle ])


dead_reckoning = np.array(dead_reckoning)


fig, ax = plt.subplots()

ax.scatter(odom[:, 0], odom[:, 1])
ax.scatter(dead_reckoning[:, 0], dead_reckoning[:, 1])

np.save("odom.npy", odom)
np.save("control.npy", control)
np.save("dead_reckoning.npy", dead_reckoning)


plt.show()