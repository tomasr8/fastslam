import matplotlib.pyplot as plt
import numpy as np
import json

track = np.load("../track.npy")

odometry = np.load("odom.npy")
odometry[:, 2] += np.pi/2
odometry[:, 2] = np.arctan2(np.sin(odometry[:, 2]), np.cos(odometry[:, 2]))
control = np.load("control.npy")

plt.scatter(track[:, 0], track[:, 1], s=3)

plt.scatter(odometry[:, 0], odometry[:, 1], s=1)
plt.axis("equal")

pose = odometry[0, :3]
for u in control[:]:
    pose[2] += u[0]
    pose[2] = np.arctan2(np.sin(pose[2]), np.cos(pose[2]))

    pose[0] += np.cos(pose[2]) * u[1]
    pose[1] += np.sin(pose[2]) * u[1]

    plt.scatter([pose[0]], [pose[1]], color="purple", s=2)


print(control[:10])
print(odometry[:10])

plt.show()