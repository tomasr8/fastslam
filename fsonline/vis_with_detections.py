import matplotlib.pyplot as plt
import numpy as np
import json

def min_dist(arr, stamp):
    arr = np.abs(arr - stamp)
    return np.argmin(arr)


track = np.load("track.npy")

with open("full_pipeline/detections_converted.json") as f:
    detections = json.load(f)

odom = np.load("full_pipeline/odom.npy")


fig, ax = plt.subplots()
ax.scatter(track[:, 0], track[:, 1])

# for d in detections:
#     stamp = d['stamp']

#     o = odom[min_dist(odom[:, -1], stamp)]
#     print(stamp, o[-1])

#     points = np.array(d['points'])[:, :2]

#     points[:, [0, 1]] = points[:, [1, 0]]
#     points[:, 1] +=2
#     points[:, 0] *= -1

#     ax.scatter(points[:, 0], points[:, 1], c="g")
#     ax.scatter([o[0]], [o[1]], c="r")
#     plt.pause(0.0001)


stamps = np.array([d['stamp'] for d in detections])

for o in odom:
    stamp = o[-1]

    d = detections[min_dist(stamps, stamp)]
    print(stamp, d['stamp'])

    points = np.array(d['points'])[:, :2]

    points[:, [0, 1]] = points[:, [1, 0]]
    points[:, 1] +=2
    points[:, 0] *= -1

    ax.scatter(points[:, 0], points[:, 1], c="g")
    ax.scatter([o[0]], [o[1]], c="r")
    plt.pause(0.0001)
