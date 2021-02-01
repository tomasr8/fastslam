import math
import time
import numpy as np
import matplotlib.pyplot as plt

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.driver import limit

MAX_BLOCK_SIZE = 1024

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

def is_valid_permutation(ancestors):
    N = len(ancestors)
    offsprings = [0] * N

    for a in ancestors:
        offsprings[a] += 1

    for i in range(N):
        if offsprings[i] > 0 and ancestors[i] != i:
            return False

    return True


def run_custom(module, ancestors, cuda_ancestors, cuda_aux):
    N = ancestors.size
    # print(ancestors)
    cuda.memcpy_htod(cuda_ancestors, ancestors)

    module.get_function("reset")(cuda_aux, block=(MAX_BLOCK_SIZE, 1, 1), grid=(N//MAX_BLOCK_SIZE, 1, 1))
    module.get_function("compute_positions")(cuda_ancestors, cuda_aux, block=(MAX_BLOCK_SIZE, 1, 1), grid=(N//MAX_BLOCK_SIZE, 1, 1))
    # aux = np.zeros(N, dtype=np.int32)
    # cuda.memcpy_dtoh(aux, cuda_aux)
    # print("aux:", aux)
    module.get_function("permute_custom")(cuda_ancestors, cuda_aux, np.int32(N//MAX_BLOCK_SIZE), np.int32(N), block=(MAX_BLOCK_SIZE, 1, 1))

    out = np.zeros_like(ancestors)
    cuda.memcpy_dtoh(out, cuda_ancestors)
    # print(out)

    # aux = np.zeros(N, dtype=np.int32)
    # cuda.memcpy_dtoh(aux, cuda_aux)
    # print("aux:", aux)

    print(is_valid_permutation(out), np.array_equal(ancestors, np.sort(out)))


def run_custom2(module, ancestors, cuda_ancestors, cuda_aux, cuda_end):
    N = ancestors.size
    # print(ancestors)
    cuda.memcpy_htod(cuda_ancestors, ancestors)

    module.get_function("reset")(cuda_aux, block=(MAX_BLOCK_SIZE, 1, 1), grid=(N//MAX_BLOCK_SIZE, 1, 1))
    module.get_function("compute_positions")(cuda_ancestors, cuda_aux, block=(MAX_BLOCK_SIZE, 1, 1), grid=(N//MAX_BLOCK_SIZE, 1, 1))
    # aux = np.zeros(N, dtype=np.int32)
    # cuda.memcpy_dtoh(aux, cuda_aux)
    # print("aux:", aux)
    module.get_function("compute_end")(cuda_ancestors, cuda_aux, cuda_end, block=(MAX_BLOCK_SIZE, 1, 1), grid=(N//MAX_BLOCK_SIZE, 1, 1))
    module.get_function("permute_custom")(cuda_ancestors, cuda_aux, cuda_end, block=(MAX_BLOCK_SIZE, 1, 1), grid=(N//MAX_BLOCK_SIZE, 1, 1))

    out = np.zeros_like(ancestors)
    cuda.memcpy_dtoh(out, cuda_ancestors)
    # print(out)

    # aux = np.zeros(N, dtype=np.int32)
    # cuda.memcpy_dtoh(aux, cuda_aux)
    # print("aux:", aux)

    # end = np.zeros(N, dtype=np.int32)
    # cuda.memcpy_dtoh(end, cuda_end)
    # print("end:", end)
    print(is_valid_permutation(out), np.array_equal(ancestors, np.sort(out)))


def run_reference(module, ancestors, cuda_ancestors, cuda_c, cuda_d):
    N = ancestors.size

    cuda.memcpy_htod(cuda_ancestors, ancestors)

    module.get_function("reset")(cuda_d, np.int32(N), block=(MAX_BLOCK_SIZE, 1, 1), grid=(N//MAX_BLOCK_SIZE, 1, 1))
    module.get_function("prepermute")(cuda_ancestors, cuda_d, block=(MAX_BLOCK_SIZE, 1, 1), grid=(N//MAX_BLOCK_SIZE, 1, 1))
    # module.get_function("permute_reference")(cuda_ancestors, cuda_c, cuda_d, np.int32(N//MAX_BLOCK_SIZE), np.int32(N), block=(MAX_BLOCK_SIZE, 1, 1))
    module.get_function("permute_reference")(cuda_ancestors, cuda_c, cuda_d, np.int32(N), block=(MAX_BLOCK_SIZE, 1, 1), grid=(N//MAX_BLOCK_SIZE, 1, 1))
    module.get_function("write_to_c")(cuda_ancestors, cuda_c, cuda_d, block=(MAX_BLOCK_SIZE, 1, 1), grid=(N//MAX_BLOCK_SIZE, 1, 1))


    out = np.zeros_like(ancestors)
    cuda.memcpy_dtoh(out, cuda_c)

    # print(is_valid_permutation(out), np.array_equal(ancestors, np.sort(out)))



if __name__ == "__main__":
    np.random.seed(0)

    context.set_limit(limit.MALLOC_HEAP_SIZE, 100000 * 1024)  # heap size available to the GPU threads

    with open("reference.cu") as f:
        reference = SourceModule(f.read())

    with open("custom.cu") as f:
        custom = SourceModule(f.read())

    with open("custom2.cu") as f:
        custom2 = SourceModule(f.read())

    # weights = np.load("../weights.npy")
    # print(weights.shape)
    # N = weights.shape[1]
    # cuda_ancestors = cuda.mem_alloc(4 * N)
    # cuda_aux = cuda.mem_alloc(4 * N)
    # cuda_end = cuda.mem_alloc(4 * N)
    # cuda_c = cuda.mem_alloc(4 * N)
    # cuda_d = cuda.mem_alloc(4 * N)

    # for i in range(weights.shape[0]):
    #     ancestors = systematic_resample(weights[i])
    #     ancestors = np.sort(ancestors)

    #     # run_custom(custom, ancestors, cuda_ancestors, cuda_aux)
    #     run_custom2(custom2, ancestors, cuda_ancestors, cuda_aux, cuda_end)
    #     run_reference(reference, ancestors, cuda_ancestors, cuda_c, cuda_d)


    N = 1024 * 128

    cuda_ancestors = cuda.mem_alloc(4 * N)
    cuda_aux = cuda.mem_alloc(4 * N)
    cuda_end = cuda.mem_alloc(4 * N)
    cuda_c = cuda.mem_alloc(4 * N)
    cuda_d = cuda.mem_alloc(4 * N)

    for i in range(100):
        # ancestors = np.random.binomial(N, 0.5, size=N).astype(np.int32)
        ancestors = np.random.randint(0, N, size=N).astype(np.int32)
        ancestors = np.sort(ancestors)

        run_custom2(custom2, ancestors, cuda_ancestors, cuda_aux, cuda_end)
        run_reference(reference, ancestors, cuda_ancestors, cuda_c, cuda_d)

    # run_custom(custom, ancestors, cuda_ancestors, cuda_aux)
    # print("====")
    # run_custom2(custom2, ancestors, cuda_ancestors, cuda_aux, cuda_end)
