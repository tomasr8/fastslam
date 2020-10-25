import math
import time
from typing import List
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

from multiprocessing import Process

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.driver import limit

from update import cuda_update
from particle import FlatParticle


cuda_time = []
cuda_htod = []
cuda_dtoh = []


def update(particles, z_real, observation_variance, cuda_particles, cuda_measurements, cuda_cov):
    measurements = z_real.astype(np.float32)

    measurement_cov = np.float32([
        observation_variance[0], 0,
        0, observation_variance[1]
    ])

    start=cuda.Event()
    end=cuda.Event()
    start.record()

    #Copies the memory from CPU to GPU
    # start = time.time()
    cuda.memcpy_htod(cuda_particles, particles)
    cuda.memcpy_htod(cuda_measurements, measurements)
    cuda.memcpy_htod(cuda_cov, measurement_cov)
    # cuda_htod.append(time.time() - start)

    threshold = 0.1

    # start = time.time()
    func = cuda_update.get_function("update")
    func(cuda_particles, cuda_measurements,
        np.int32(FlatParticle.len(particles)), np.int32(len(measurements)),
        cuda_cov, np.float32(threshold), block=(256, 1, 1), grid=(1, 1, 1))
    # cuda_time.append(time.time() - start)

    # start = time.time()
    # particles = np.empty_like(particles)

    cuda.memcpy_dtoh(particles, cuda_particles)

    end.record()
    end.synchronize()
    cuda_time.append(start.time_till(end))
    # cuda_dtoh.append(time.time() - start)

    # start = time.time()
    # FlatParticle.rescale(particles)
    # cuda_time.append(time.time() - start)


    # print("rescale took:", time.time() - start)

    return particles


if __name__ == "__main__":
    context.set_limit(limit.MALLOC_HEAP_SIZE, 100000 * 1024)

    np.random.seed(2)

    NL = 10

    landmarks = np.zeros((NL, 2), dtype=np.float32)
    landmarks[:, 0] = np.random.uniform(0, 1, NL)
    landmarks[:, 1] = np.random.uniform(0, 1, NL)

    N = 256
    MAX_LANDMARKS = 250
    particles = FlatParticle.get_initial_particles(N, MAX_LANDMARKS, [0, 0, 0], sigma=0.2)

    cuda_particles = cuda.mem_alloc(4 * N * (6 + 6*MAX_LANDMARKS))
    cuda_measurements = cuda.mem_alloc(1024)
    cuda_cov = cuda.mem_alloc(32)

    measurement_variance = [0.2, 0.2]

    FlatParticle.set_lm(particles, landmarks, np.diag(measurement_variance))

    NM = 20
    for i in range(60):
        print(i)

        measurements = np.zeros((NM, 2), dtype=np.float32)
        measurements[:, 0] = np.random.uniform(0, 1, NM)
        measurements[:, 1] = np.random.uniform(0, 1, NM)
        particles = update(particles, measurements, measurement_variance, cuda_particles, cuda_measurements, cuda_cov)


    # print("Mean HTOD time: ", np.mean(cuda_htod))
    print("Mean CUDA time: ", np.mean(cuda_time) / 1000, np.std(cuda_time) / 1000)
    print(cuda_time)
    # print("Mean DTOH time: ", np.mean(cuda_dtoh))

    # print(cuda_time)