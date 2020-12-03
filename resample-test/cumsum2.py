import numpy as np
import time
import math
import matplotlib.pyplot as plt

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.driver import limit

context.set_limit(limit.MALLOC_HEAP_SIZE, 100000 * 1024)

with open("cumsum.cu") as f:
        source = f.read()

cuda_get_cumsum = SourceModule(source)


# https://github.com/gggmmst/cuda-parallel-psum

def cumsum_prepare(N, THREADS_PER_BLOCK):
    nthreads = THREADS_PER_BLOCK
    block_size = 2 * nthreads
    smem = block_size * 4

    if N % block_size == 0:
        n = N
    else:
        n = (1+N//block_size)*block_size

    nblocks = n//block_size

    cuda_d_in = cuda.mem_alloc(4 * N)
    cuda_d_scan = cuda.mem_alloc(4 * n)
    cuda_d_sums = cuda.mem_alloc(4 * nblocks)
    cuda_d_incr = cuda.mem_alloc(4 * nblocks)

    return cuda_d_in, cuda_d_scan, cuda_d_sums, cuda_d_incr


def compute_cumsum(weights, cuda_d_in, cuda_d_scan, cuda_d_sums, cuda_d_incr, THREADS_PER_BLOCK):
    N = weights.shape[0]
    nthreads = THREADS_PER_BLOCK
    block_size = 2 * nthreads
    smem = block_size * 4

    if N % block_size == 0:
        n = N
    else:
        n = (1+N//block_size)*block_size

    nblocks = n//block_size

    cuda.memcpy_htod(cuda_d_in, weights)

    cuda_get_cumsum.get_function("block_psum")(
        cuda_d_in, cuda_d_scan, cuda_d_sums,
        np.int32(block_size), np.int32(1),
        grid=(nblocks, 1, 1), block=(nthreads, 1, 1), shared=smem
    )


    cuda_get_cumsum.get_function("block_psum")(
        cuda_d_sums, cuda_d_incr, cuda_d_sums,
        np.int32(block_size), np.int32(0),
        grid=(1, 1, 1), block=(nthreads, 1, 1), shared=smem
    )


    cuda_get_cumsum.get_function("scatter_incr")(
        cuda_d_scan, cuda_d_incr,
        grid=(nblocks, 1, 1), block=(nthreads, 1, 1)
    )


    out = np.zeros(N).astype(np.float32)
    cuda.memcpy_dtoh(out, cuda_d_scan)
    return out


N = 1024 * 77
THREADS_PER_BLOCK = 1024
weights = np.random.uniform(0, 1, N).astype(np.float32)
weights /= np.sum(weights)

cuda_d_in, cuda_d_scan, cuda_d_sums, cuda_d_incr = cumsum_prepare(weights.shape[0], THREADS_PER_BLOCK)
out = compute_cumsum(weights, cuda_d_in, cuda_d_scan, cuda_d_sums, cuda_d_incr, THREADS_PER_BLOCK)

print(out.shape)

print(out - np.cumsum(weights))

plt.plot(out)
plt.plot(np.cumsum(weights))
plt.show()


# MAX_THREADS_PER_BLOCK = 1024
# N = 8192 * 16
# weights = np.random.uniform(0, 1, N).astype(np.float32)
# weights /= np.sum(weights)


# nthreads = MAX_THREADS_PER_BLOCK
# block_size = 2 * nthreads
# smem = block_size * 4

# # n = smallest multiple of block_size such that larger than or equal to len
# if N % block_size == 0:
#     n = N
# else:
#     n = (1+N//block_size)*block_size

# nblocks = n//block_size


# cuda_d_in = cuda.mem_alloc(4 * N)
# cuda.memcpy_htod(cuda_d_in, weights)


# cuda_d_scan = cuda.mem_alloc(4 * n)
# cuda_d_sums = cuda.mem_alloc(4 * nblocks)
# cuda_d_incr = cuda.mem_alloc(4 * nblocks)

# cuda_get_cumsum.get_function("block_psum")(
#     cuda_d_in, cuda_d_scan, cuda_d_sums,
#     np.int32(block_size), np.int32(1),
#     grid=(nblocks, 1, 1), block=(nthreads, 1, 1), shared=smem
# )


# cuda_get_cumsum.get_function("block_psum")(
#     cuda_d_sums, cuda_d_incr, cuda_d_sums,
#     np.int32(block_size), np.int32(0),
#     grid=(1, 1, 1), block=(nthreads, 1, 1), shared=smem
# )


# cuda_get_cumsum.get_function("scatter_incr")(
#     cuda_d_scan, cuda_d_incr,
#     grid=(nblocks, 1, 1), block=(nthreads, 1, 1)
# )


# out = np.zeros(n).astype(np.float32)
# cuda.memcpy_dtoh(out, cuda_d_scan)

# print(out.shape)

# print(out - np.cumsum(weights))

# plt.plot(out)
# plt.plot(np.cumsum(weights))
# plt.show()