import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import json
from plotting import (
    plot_connections, plot_history, plot_landmarks, plot_measurement,
    plot_particles_weight, plot_particles_grey, plot_confidence_ellipse,
    plot_sensor_fov, plot_map
)

def to_coords(range, bearing, theta):
    return [range * np.cos(bearing + theta), range * np.sin(bearing + theta)]


START = 1000
T = 1000.0
robot = 2

ground = np.load(f"utias/npy/ground_{robot}_50hz.npy")[START:]
measurements = np.load(f"utias/npy/measurements_xy_{robot}_50hz.npy")
measurements_rb = np.load(f"utias/npy_fixed/measurements_rb_{robot}_50hz.npy")
landmarks = np.load("utias/landmarks.npy")[:, 1:]

ground = ground[ground[:, 0] < T]
measurements = measurements[measurements[:, 0] < T]
measurements_rb = measurements_rb[measurements_rb[:, 0] < T]


fig, ax = plt.subplots()

# ground = ground[::100]
# predicted = np.array(output["predicted"][::100])

print(ground.shape)

for i in range(ground.shape[0]):
    if i % 1000 == 0:
        print(i)
    ax.clear()
    plot_history(ax, ground[:i, 1:], color='green')

    t = ground[i, 0]

    m = measurements[measurements[:, 0] == t]
    m = m[:, [2,3]].astype(np.float32)

    m_rb = measurements_rb[measurements_rb[:, 0] == t]
    m_rb = m_rb[:, [2,3]].astype(np.float32)
    m_rb = [to_coords(r, b, ground[i, 3]) for r, b in m_rb]
    m_rb = np.array(m_rb)

    if m.size != 0 and m_rb.size != 0:
        print(m)
        print(m_rb)
        plot_measurement(ax, ground[i, 1:3], m, color="orange", zorder=103, size=60)
        plot_measurement(ax, ground[i, 1:3], m_rb, color="red", zorder=104)
        plot_landmarks(ax, landmarks, color="blue")
        plt.pause(0.5)
    else:
        plot_landmarks(ax, landmarks, color="blue")
    # if m_rb.size != 0:
    #     plot_landmarks(ax, landmarks, color="blue")
    #     plt.pause(2)

    plt.pause(0.01)


plt.show()