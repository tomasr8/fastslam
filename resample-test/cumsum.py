import numpy as np
import time
import math

# N = 8192 * 8
# arr = np.random.uniform(0, 1, size=N).astype(np.float32)
# out = np.zeros(N+1, dtype=np.float32)

# for _ in range(20):
#     start = time.time()
#     np.cumsum(arr, out=out[1:])
#     print(time.time() - start)
#     print(out[:10])



def systematic(weights, cumsum, indices, rand):
    N = indices.shape[0]


    for k in range(N):
        # left = math.floor((cumsum[k] - rand) * N) + 1
        # right = math.floor((cumsum[k] + weights[k] - rand) * N)

        # print(k, left, right)
        # left = math.ceil((cumsum[k] * N) - rand)
        # right =  math.ceil(((cumsum[k] + weights[k]) * N) - rand)

        left = math.ceil(((cumsum[k] - weights[k]) * N) - rand)
        right = math.ceil((cumsum[k] * N) - rand)
        # print(k, "->", left, right)

        for j in range(left, right):
            indices[j] = k


# weights = np.array([0.1, 0.4, 0.2, 0.1, 0.2], dtype=np.float32)
# cumsum = np.zeros(weights.shape[0] + 1, dtype=np.float32)
# cumsum[1:] = np.cumsum(weights)
# cumsum = cumsum[:-1]

N = 8192
weights = np.random.uniform(0, 1, N).astype(np.float32)
weights /= np.sum(weights)
indices = np.zeros(weights.shape[0], dtype=np.int32)

# np.random.seed(0)
# rand = np.random.uniform()
rand = 0.4

print(rand)
cumsum = np.cumsum(weights)
cumsum[-1] = 1.0
systematic(weights, cumsum, indices, rand)
# systematic(weights, cumsum, indices, rand)

print(indices)