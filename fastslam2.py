import math
import time
from typing import List
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

from plotting import (
    plot_connections, plot_history, plot_landmarks, plot_measurement,
    plot_particles_weight, plot_particles_grey, plot_confidence_ellipse,
    plot_sensor_fov, plot_map
)
from particle import Particle, FlatParticle, systematic_resample
from utils import dist

from multiprocessing import Process

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.driver import limit

from cuda.update2 import cuda_update, cuda_resample, cuda_predict, cuda_mean
from sensor import Sensor
from map import compute_map

cuda_time = []
cuda_resample_time = []


class Vehicle(object):
    def __init__(self, position, movement_variance, dt):
        self.position = position
        self.movement_variance = movement_variance
        self.dt = dt
        self.random = np.random.RandomState(seed=4)

    def move_noisy(self, u):
        '''Stochastically moves the vehicle based on the control input and noise
        '''
        x, y, theta = self.position

        if u[0] == 0.0 and u[1] == 0.0:
            return

        theta += (u[0] + self.random.normal(0, self.movement_variance[0]))
        theta %= (2*math.pi)

        dist = (u[1] * self.dt) + self.random.normal(0, self.movement_variance[1])
        x += np.cos(theta) * dist
        y += np.sin(theta) * dist

        self.position = np.array([x, y, theta]).astype(np.float32)


def mean_path_deviation(real_path, predicted_path):
    diffs = []

    for real, predicted in zip(real_path, predicted_path):
        x1, y1, _ = real
        x2, y2, _ = predicted
        diffs.append([
            abs(x1-x2),
            abs(y1-y2)
        ])

    return np.array(diffs).mean(axis=0)

def update(
        i, particles, threads, block_size, z_real, observation_variance, cuda_particles, cuda_new_particles,
        cuda_idx, cuda_measurements, cuda_cov, threshold, max_range, max_fov, u, sigmas, dt):



    measurements = z_real.astype(np.float32)

    measurement_cov = np.float32([
        observation_variance[0], 0,
        0, observation_variance[1]
    ])

    start = cuda.Event()
    end = cuda.Event()
    start.record()

    cuda.memcpy_htod(cuda_particles, particles)
    cuda.memcpy_htod(cuda_new_particles, particles)
    cuda.memcpy_htod(cuda_measurements, measurements)
    cuda.memcpy_htod(cuda_cov, measurement_cov)

    if i == 0:
        func = cuda_predict.get_function("init_kernel")
        func(np.int32(2), block=(threads, 1, 1))

    func = cuda_predict.get_function("predict")
    func(
        cuda_particles, np.int32(block_size), np.int32(FlatParticle.len(particles)),
        np.float32(u[0]), np.float32(u[1]), np.float32(sigmas[0]), np.float32(sigmas[1]),
        np.float32(dt),
        block=(threads, 1, 1)
    )

    func = cuda_update.get_function("update")
    func(cuda_particles, cuda_new_particles, np.int32(block_size), cuda_measurements,
         np.int32(FlatParticle.len(particles)), np.int32(len(measurements)),
         cuda_cov, np.float32(threshold), np.float32(max_range), np.float32(max_fov),
         block=(threads, 1, 1), grid=(1, 1, 1)
         )

    # cuda.memcpy_dtoh(particles, cuda_new_particles)
    cuda.memcpy_dtoh(particles, cuda_particles)


    end.record()
    end.synchronize()
    cuda_time.append(start.time_till(end))

    # ======================================

    # start = time.time()

    # N = FlatParticle.len(particles)
    # max_landmarks = int(particles[4])
    # size = 6 + 7*max_landmarks

    # weights = FlatParticle.w(particles)
    # indexes = systematic_resample(weights).astype(np.int32)

    # cuda.memcpy_htod(cuda_idx, indexes)

    # if FlatParticle.neff(particles) < 0.6*FlatParticle.len(particles):
    #     func = cuda_resample.get_function("resample")
    #     func(
    #         cuda_particles,
    #         cuda_new_particles, cuda_idx, np.int32(block_size),
    #         np.int32(FlatParticle.len(particles)), 
    #         # np.float32(np.random.uniform()),
    #         block=(threads, 1, 1), grid=(1, 1, 1)
    #     )

    # w = np.zeros(FlatParticle.len(particles), dtype=np.float32)
    # cuda.memcpy_dtoh(w, cuda_idx)

    # cuda_resample_time.append((time.time() - start)*1000)


    FlatParticle.rescale(particles)
    cuda.memcpy_htod(cuda_particles, particles)




    func = cuda_mean.get_function("mean_position")
    func(
        cuda_particles, np.int32(FlatParticle.len(particles)),
        block=(1, 1, 1)
    )

    # max_landmarks = int(particles[4])
    # step = 6 + 7*max_landmarks

    # xs = particles[0::step]

    # print("xs sum", np.sum(xs))
    print("mean", FlatParticle.get_mean_position(particles))

    return particles


if __name__ == "__main__":
    # randomness
    np.random.seed(2)

    # visualization
    PLOT = False

    # simulation
    N = 256  # number of particles
    SIM_LENGTH = 25  # number of simulation steps
    MAX_RANGE = 5  # max range of sensor
    MAX_FOV = (1)*np.pi
    DT = 0.5
    MISS_PROB = 0.05  # probability landmark in range will be missed
    MAX_LANDMARKS = 150  # upper bound on the total number of landmarks in the environment
    MAX_MEASUREMENTS = 50  # upper bound on the total number of simultaneous measurements
    landmarks = np.loadtxt("landmarks.txt").astype(np.float32)  # landmark positions
    start_position = np.array([8, 3, 0], dtype=np.float32)  # starting position of the car
    movement_variance = [0.07, 0.07]
    measurement_variance = [0.1, 0.1]
    THRESHOLD = 0.2

    # GPU
    context.set_limit(limit.MALLOC_HEAP_SIZE, 100000 * 1024)  # heap size available to the GPU threads
    THREADS = 256  # number of GPU threads
    assert N >= THREADS
    BLOCK_SIZE = N//THREADS  # number of particles per thread

    particles = FlatParticle.get_initial_particles(N, MAX_LANDMARKS, start_position, sigma=0.2)
    print("Particles memory:", particles.nbytes / 1024, "KB")

    if PLOT:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_xlim([-5, 20])
        ax[0].set_ylim([-5, 20])
        ax[1].set_xlim([-5, 20])
        ax[1].set_ylim([-5, 20])
        ax[0].axis('scaled')
        ax[1].axis('scaled')


    u = np.vstack((
        np.tile([0.0, 0.0], (5, 1)),
        np.tile([0.06, 0.7], (SIM_LENGTH-5, 1))
    ))

    real_position_history = [start_position]
    predicted_position_history = [FlatParticle.get_mean_position(particles)]
    weights_history = [FlatParticle.neff(particles)]

    vehicle = Vehicle(start_position, movement_variance, dt=DT)
    sensor = Sensor(vehicle, landmarks, [], measurement_variance, MAX_RANGE, MAX_FOV, MISS_PROB, 0)

    cuda_particles = cuda.mem_alloc(4 * N * (6 + 7*MAX_LANDMARKS))
    cuda_new_particles = cuda.mem_alloc(4 * N * (6 + 7*MAX_LANDMARKS))
    cuda_idx = cuda.mem_alloc(4 * N)
    # cuda_weights = cuda.mem_alloc(4 * N)
    cuda_measurements = cuda.mem_alloc(4 * 2 * MAX_MEASUREMENTS)
    cuda_cov = cuda.mem_alloc(4 * 4)

    # plt.pause(5)

    for i in range(u.shape[0]):
        print(i)

        vehicle.move_noisy(u[i])
        real_position_history.append(vehicle.position)

        # start = time.time()
        # FlatParticle.predict(particles, u[i], sigmas=movement_variance, dt=DT)
        # print("predict: ", time.time() - start)

        measurements = sensor.get_noisy_measurements()
        visible_measurements = measurements["observed"]

        # visible_measurements = np.vstack((measurements["observed"], measurements["phantomSeen"]))
        missed_landmarks = measurements["missed"]
        out_of_range_landmarks = measurements["outOfRange"]
        # visible_measurements = sensor.get_noisy_measurements(vehicle.position[:2])

        particles = update(
            i, particles, THREADS, BLOCK_SIZE, visible_measurements, measurement_variance,
            cuda_particles, cuda_new_particles, cuda_idx, cuda_measurements, cuda_cov, THRESHOLD, MAX_RANGE, MAX_FOV,
            u[i], movement_variance, DT
        )

        predicted_position_history.append(FlatParticle.get_mean_position(particles))

        if PLOT:
            ax[0].clear()
            ax[0].set_xlim([-5, 20])
            ax[0].set_ylim([-5, 20])

            plot_sensor_fov(ax[0], vehicle, MAX_RANGE, MAX_FOV)
            plot_sensor_fov(ax[1], vehicle, MAX_RANGE, MAX_FOV)

            if(visible_measurements.size != 0):
                plot_connections(ax[0], vehicle.position, visible_measurements + vehicle.position[:2])

            plot_landmarks(ax[0], landmarks, color="blue", zorder=100)
            plot_landmarks(ax[0], out_of_range_landmarks, color="black", zorder=101)
            plot_history(ax[0], real_position_history, color='green')
            plot_history(ax[0], predicted_position_history, color='orange')
            plot_particles_weight(ax[0], particles)
            if(visible_measurements.size != 0):
                plot_measurement(ax[0], vehicle.position[:2], visible_measurements, color="orange", zorder=103)

            plot_landmarks(ax[0], missed_landmarks, color="red", zorder=102)

            ax[1].clear()
            ax[1].set_xlim([-5, 20])
            ax[1].set_ylim([-5, 20])
            # best = np.argmax(FlatParticle.w(particles))
            plot_landmarks(ax[1], landmarks, color="black")
            # plot_landmarks(ax[1], FlatParticle.get_landmarks(particles, best), color="orange")
            # covariances = FlatParticle.get_covariances(particles, best)

            # for i, landmark in enumerate(FlatParticle.get_landmarks(particles, best)):
            #     plot_confidence_ellipse(ax[1], landmark, covariances[i], n_std=3)


            # n = 200
            # cmap = plt.cm.get_cmap("hsv", n)
            # print("n_landmarks:", [len(FlatParticle.get_landmarks(particles, i)) for i in np.argsort(-FlatParticle.w(particles))[:n]])
            # for n, i in enumerate(np.argsort(-FlatParticle.w(particles))[:n]):
            #     plot_map(ax[1], FlatParticle.get_landmarks(particles, i), color=cmap(n), marker=".")

            # start = time.time()
            # centroids = compute_map(particles)
            # print("kmeans: ", time.time() - start)
            # plot_map(ax[1], centroids, color="orange", marker="o")

            plt.pause(0.2)


        if FlatParticle.neff(particles) < 0.6*N:
            print("resampling..")
            start = time.time()
            particles = FlatParticle.resample2(particles)
            print("resample: ", time.time() - start)

        weights_history.append(FlatParticle.neff(particles))


    print("Mean CUDA compute time: ", np.mean(cuda_time) / 1000, ", stdev: ", np.std(cuda_time) / 1000)
    # print("Mean CUDA resample time: ", np.mean(cuda_resample_time) / 1000, ", stdev: ", np.std(cuda_resample_time) / 1000)
    deviation = mean_path_deviation(real_position_history, predicted_position_history)
    print(f"Mean path deviation: {deviation}")