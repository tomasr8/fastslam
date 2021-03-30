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

with open("out3.json") as f:
    output = json.load(f)


fig, ax = plt.subplots()

ground = np.array(output["ground"][::200])
predicted = np.array(output["predicted"][::200])

print(ground.shape)

for i in range(51, ground.shape[0]):
    ax.clear()
    plot_history(ax, ground[i-50:i], color='green')
    plot_history(ax, predicted[i-50:i], color='orange')
    plot_landmarks(ax, np.array(output["landmarks"]), color="blue")
    plt.pause(0.01)


plt.show()