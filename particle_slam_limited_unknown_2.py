import math
import time
from typing import List
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
from filterpy.monte_carlo import systematic_resample

from plotting import plot_connections, plot_history, plot_landmarks, plot_measurement, plot_particles_weight, plot_particles_grey
from particle import Particle, FlatParticle
# from data_association import associate_landmarks_measurements
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


def get_noisy_measurement(position, landmark, measurement_variance):
    '''Returns a noisy measurement given by the variance
    '''
    vector_to_landmark = np.array(landmark - position, dtype=np.float32)

    a = np.random.normal(0, measurement_variance[0])
    vector_to_landmark[0] += a
    b = np.random.normal(0, measurement_variance[1])
    vector_to_landmark[1] += b

    return vector_to_landmark


# def get_noisy_measurements(position, landmarks, measurement_variance):
#     '''
#         Returns a noisy measurement given by the variance
#     '''
#     n_landmarks = landmarks.shape[0]
#     measurements = landmarks - position

#     measurements[:, 0] += np.random.normal(loc=0, scale=measurement_variance[0], size=n_landmarks)
#     measurements[:, 1] += np.random.normal(loc=0, scale=measurement_variance[1], size=n_landmarks)

#     return measurements


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
    cuda_measurements, cuda_cov):

    measurements = z_real.astype(np.float32)

    measurement_cov = np.float32([
        observation_variance[0], 0,
        0, observation_variance[1]
    ])

    start=cuda.Event()
    end=cuda.Event()
    start.record()

    #Copies the memory from CPU to GPU
    cuda.memcpy_htod(cuda_particles, particles)
    cuda.memcpy_htod(cuda_measurements, measurements)
    cuda.memcpy_htod(cuda_cov, measurement_cov)

    threshold = 0.1

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
    N = 1024
    SIM_LENGTH = 100
    MAX_RANGE = 3
    DROPOUT = 1.0

    # GPU
    context.set_limit(limit.MALLOC_HEAP_SIZE, 100000 * 1024)
    THREADS = 512
    BLOCK_SIZE = N//THREADS


    mean_landmarks = []

    if PLOT:
        fig, ax = plt.subplots()
        ax.set_xlim([0, 15])
        ax.set_ylim([0, 15])

    landmarks = np.loadtxt("landmarks.txt").astype(np.float32)
    
    real_position = np.array([8, 3, 0], dtype=np.float32)

    MAX_LANDMARKS = 100
    particles = FlatParticle.get_initial_particles(N, MAX_LANDMARKS, real_position, sigma=0.2)
    print("nbytes", particles.nbytes)

    u = np.vstack((
        np.tile([0.13, 0.7], (SIM_LENGTH, 1))
    ))

    real_position_history = [real_position]
    predicted_position_history = [FlatParticle.get_mean_position(particles)]

    movement_variance = [0.03, 0.05]
    measurement_variance = [0.1, 0.1]
    sensor = Sensor(landmarks, measurement_variance, MAX_RANGE, DROPOUT)

    cuda_particles = cuda.mem_alloc(4 * N * (6 + 6*MAX_LANDMARKS))
    cuda_measurements = cuda.mem_alloc(1024)
    cuda_cov = cuda.mem_alloc(16)

    for i in range(u.shape[0]):
        print(i)

        if PLOT:
            plt.pause(0.05)
            ax.clear()
            ax.set_xlim([0, 15])
            ax.set_ylim([0, 15])
            plot_landmarks(ax, landmarks)
            plot_history(ax, real_position_history, color='green')
            plot_history(ax, predicted_position_history, color='orange')
            plot_particles_grey(ax, particles)


        real_position = move_vehicle_noisy(real_position, u[i], dt=1, sigmas=movement_variance)
        real_position_history.append(real_position)

        FlatParticle.predict(particles, u[i], sigmas=movement_variance, dt=1)

        visible_measurements = []
        for i, landmark in enumerate(landmarks):
            z = get_noisy_measurement(real_position[:2], landmark, measurement_variance)

            if dist(landmark, real_position) < MAX_RANGE:
                visible_measurements.append(z)

        visible_measurements = np.array(visible_measurements, dtype=np.float32)

        particles = update(
            particles, THREADS, BLOCK_SIZE, visible_measurements, measurement_variance,
            cuda_particles,
            cuda_measurements, cuda_cov
        )

        # plt.pause(2)

        predicted_position_history.append(FlatParticle.get_mean_position(particles))

        if PLOT:
            ax.clear()
            ax.set_xlim([0, 15])
            ax.set_ylim([0, 15])
            plot_landmarks(ax, landmarks)
            plot_history(ax, real_position_history, color='green')
            plot_history(ax, predicted_position_history, color='orange')
            plot_connections(ax, real_position, visible_measurements + real_position[:2])
            plot_particles_weight(ax, particles)
            plot_measurement(ax, real_position[:2], visible_measurements, color="red")

        if FlatParticle.neff(particles) < 0.6*N:
            print("resampling..")
            particles = FlatParticle.resample(particles)

    print("Mean CUDA compute time: ", np.mean(cuda_time) / 1000, ", stdev: ", np.std(cuda_time) / 1000)