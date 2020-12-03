import math
import time
import numpy as np
import matplotlib.pyplot as plt


import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.driver import limit



if __name__ == "__main__":
    context.set_limit(limit.MALLOC_HEAP_SIZE, 100000 * 1024)  # heap size available to the GPU threads

    with open("resample.cu") as f:
        source = f.read()

    cuda_get_sum = SourceModule(source, no_extern_c=True)
    cuda_sum = cuda.mem_alloc(4)

    N = 8192 * 4
    THREADS = 1024
    indices = np.zeros(N, dtype=np.float32)
    cuda_indices = cuda.mem_alloc(4 * N)
    cuda_cumsum = cuda.mem_alloc(4 * N)
    cuda.memcpy_htod(cuda_indices, indices)


    cuda_get_sum.get_function("init_rng")(
        np.int32(0),
        block=(THREADS, 1, 1)
    )

    for _ in range(10):
        weights = np.random.uniform(0, 1, N).astype(np.float32)
        weights /= np.sum(weights)
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0
        m = np.max(weights)

        cuda_weights = cuda.mem_alloc(weights.nbytes)
        cuda.memcpy_htod(cuda_weights, weights)
        cuda.memcpy_htod(cuda_cumsum, cumsum)

        # cuda_get_sum.get_function("metropolis")(
        #     cuda_weights, np.int32(N), cuda_indices,
        #     block=(THREADS, 1, 1)
        # )

        # cuda_get_sum.get_function("rejection")(
        #     cuda_weights, np.int32(N), np.float32(m), cuda_indices,
        #     block=(THREADS, 1, 1)
        # )

        # rand = np.random.uniform()
        rand = 0.2

        cuda_get_sum.get_function("systematic")(
            cuda_weights, cuda_cumsum, np.int32(N), np.float32(rand), cuda_indices,
            block=(THREADS, 1, 1)
        )

        s = np.zeros(1, dtype=np.float32)
        cuda.memcpy_dtoh(indices, cuda_indices)


    # for i in range(1, 7):
    #     size = 10**i
    #     weights = np.random.uniform(0, 1, size).astype(np.float32)

    #     cuda_mem = cuda.mem_alloc(weights.nbytes)
    #     cuda.memcpy_htod(cuda_mem, weights)

    #     cuda_get_sum.get_function("sum_weights")(
    #         cuda_mem, np.int32(size), cuda_sum,
    #         block=(512, 1, 1)
    #     )

    #     s = np.zeros(1, dtype=np.float32)
    #     cuda.memcpy_dtoh(s, cuda_sum)
