import math
import time
from typing import List
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

from plotting import plot_connections, plot_history, plot_landmarks, plot_measurement, plot_particles_weight, plot_particles_grey
from particle import Particle, FlatParticle
from utils import dist

from multiprocessing import Process

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.driver import limit

from cuda.update import cuda_update

cuda_time = []

class Sensor(object):
    def __init__(self, landmarks, measurement_variance, max_range, dropout):
        self.landmarks = landmarks
        self.measurement_variance = measurement_variance
        self.max_range = max_range
        self.dropout = dropout

    def __get_noisy_measurement(self, position, landmark, measurement_variance):
        vector_to_landmark = np.array(landmark - position, dtype=np.float32)

        a = np.random.normal(0, measurement_variance[0])
        vector_to_landmark[0] += a
        b = np.random.normal(0, measurement_variance[1])
        vector_to_landmark[1] += b

        return vector_to_landmark

    def get_noisy_measurements(self, position):
        visible_measurements = []
        for i, landmark in enumerate(landmarks):
            z = self.__get_noisy_measurement(position, landmark, measurement_variance)

            coin_toss = np.random.uniform(0, 1)
            if dist(landmark, position) < self.max_range and coin_toss < self.dropout:
                visible_measurements.append(z)

        return np.array(visible_measurements, dtype=np.float32)   


def mean_path_deviation(real_path, predicted_path):
    total = 0

    for real, predicted in zip(real_path, predicted_path):
        total += dist(real[:2], predicted[:2])

    return total/len(real_path)


def move_vehicle_noisy(pos, u, dt, sigmas):
    '''Stochastically moves the vehicle based on the control input and noise
    '''
    x, y, theta = pos

    theta += u[0] + np.random.normal(0, sigmas[0])

    dist = (u[1] * dt) + np.random.normal(0, sigmas[1])
    x += np.cos(theta) * dist
    y += np.sin(theta) * dist

    return np.array([x, y, theta], dtype=np.float32)


def update(
    particles, threads, block_size, z_real, observation_variance, cuda_particles,
    cuda_measurements, cuda_cov, threshold):

    measurements = z_real.astype(np.float32)

    measurement_cov = np.float32([
        observation_variance[0], 0,
        0, observation_variance[1]
    ])

    start=cuda.Event()
    end=cuda.Event()
    start.record()

    cuda.memcpy_htod(cuda_particles, particles)
    cuda.memcpy_htod(cuda_measurements, measurements)
    cuda.memcpy_htod(cuda_cov, measurement_cov)

    func = cuda_update.get_function("update")
    func(cuda_particles, np.int32(block_size), cuda_measurements,
        np.int32(FlatParticle.len(particles)), np.int32(len(measurements)),
        cuda_cov, np.float32(threshold),
        block=(threads, 1, 1), grid=(1, 1, 1)
    )

    cuda.memcpy_dtoh(particles, cuda_particles)

    end.record()
    end.synchronize()
    cuda_time.append(start.time_till(end))

    FlatParticle.rescale(particles)
    return particles


if __name__ == "__main__":
    # randomness
    np.random.seed(2)

    # visualization
    PLOT = True

    # simulation
    N = 4096 # number of particles
    SIM_LENGTH = 100 # number of simulation steps
    MAX_RANGE = 5 # max range of sensor
    DROPOUT = 1.0 # probability landmark in range will be seen
    MAX_LANDMARKS = 100 # upper bound on the total number of landmarks in the environment
    landmarks = np.loadtxt("landmarks.txt").astype(np.float32) # landmark positions
    real_position = np.array([8, 3, 0], dtype=np.float32) # starting position of the car
    movement_variance = [0.1, 0.1]
    measurement_variance = [0.1, 0.1]
    THRESHOLD = 0.2

    # GPU
    context.set_limit(limit.MALLOC_HEAP_SIZE, 100000 * 1024) # heap size available to the GPU threads
    THREADS = 512 # number of GPU threads
    BLOCK_SIZE = N//THREADS # number of particles per thread


    particles = FlatParticle.get_initial_particles(N, MAX_LANDMARKS, real_position, sigma=0.2)
    print("Particles memory:", particles.nbytes / 1024, "KB")


    if PLOT:
        fig, ax = plt.subplots()
        ax.set_xlim([0, 25])
        ax.set_ylim([0, 25])

    u = np.vstack((
        np.tile([0.13, 0.7], (SIM_LENGTH, 1))
    ))

    real_position_history = [real_position]
    predicted_position_history = [FlatParticle.get_mean_position(particles)]

    sensor = Sensor(landmarks, measurement_variance, MAX_RANGE, DROPOUT)

    cuda_particles = cuda.mem_alloc(4 * N * (6 + 6*MAX_LANDMARKS))
    cuda_measurements = cuda.mem_alloc(1024)
    cuda_cov = cuda.mem_alloc(16)

    # plt.pause(5)

    for i in range(u.shape[0]):
        print(i)

        if PLOT:
            plt.pause(0.05)
            ax.clear()
            ax.set_xlim([0, 25])
            ax.set_ylim([0, 25])
            plot_landmarks(ax, landmarks)
            plot_history(ax, real_position_history, color='green')
            plot_history(ax, predicted_position_history, color='orange')
            plot_particles_grey(ax, particles)


        real_position = move_vehicle_noisy(real_position, u[i], dt=1, sigmas=movement_variance)
        real_position_history.append(real_position)

        FlatParticle.predict(particles, u[i], sigmas=movement_variance, dt=1)

        visible_measurements = sensor.get_noisy_measurements(real_position[:2])

        particles = update(
            particles, THREADS, BLOCK_SIZE, visible_measurements, measurement_variance,
            cuda_particles, cuda_measurements, cuda_cov, THRESHOLD
        )

        # plt.pause(2)

        predicted_position_history.append(FlatParticle.get_mean_position(particles))

        if PLOT:
            ax.clear()
            ax.set_xlim([0, 25])
            ax.set_ylim([0, 25])
            plot_landmarks(ax, landmarks)
            plot_history(ax, real_position_history, color='green')
            plot_history(ax, predicted_position_history, color='orange')
            if(visible_measurements.size != 0):
                plot_connections(ax, real_position, visible_measurements + real_position[:2])
            plot_particles_weight(ax, particles)
            if(visible_measurements.size != 0):
                plot_measurement(ax, real_position[:2], visible_measurements, color="red")

        if FlatParticle.neff(particles) < 0.6*N:
            print("resampling..")
            particles = FlatParticle.resample(particles)

    print("Mean CUDA compute time: ", np.mean(cuda_time) / 1000, ", stdev: ", np.std(cuda_time) / 1000)