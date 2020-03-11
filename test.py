import numpy as np
import matplotlib.pyplot as plt


# weights = np.load("weights.npy")

# weights = sorted(weights)


# plt.plot(weights)
# plt.show()


landmarks = np.array([
    [5, 2],
    [7, 6],
    [2, 8],
    [2, 3],
    [5, 5]
])

pos = np.array([1, 1])


print(landmarks - pos)