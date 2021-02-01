import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

def systematic_resample(weights):
    N = weights.shape[0]

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (np.random.random() + np.arange(N)) / N

    indexes = np.zeros(N, dtype=np.int32)
    cumulative_sum = np.cumsum(weights)
    # prevent float imprecision
    cumulative_sum[-1] = 1.0

    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


weights = np.load("../weights.npy")
plt.plot(weights[50])
plt.show()


# plt.hist(systematic_resample(weights[50]), bins=1000)
# plt.show()

# N = 8192
# ancestors = np.round(np.random.normal(0, 1000, size=N) * N + N/2).round()
# plt.hist(ancestors)
# plt.show()