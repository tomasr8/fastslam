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
    plot_sensor_fov
)
from particle import Particle, FlatParticle
from utils import dist

from multiprocessing import Process

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.driver import limit

from cuda.update import cuda_update
from sensor import Sensor

cuda_time = []


class Vehicle(object):
    def __init__(self, position, movement_variance, dt):
        self.position = position
        self.movement_variance = movement_variance
        self.dt = dt
        self.random = np.random.RandomState(seed=6)

    def move_noisy(self, u):
        '''Stochastically moves the vehicle based on the control input and noise
        '''
        x, y, theta = self.position

        theta += (u[0] + self.random.normal(0, self.movement_variance[0]))
        theta %= (2*math.pi)

        dist = (u[1] * self.dt) + self.random.normal(0, self.movement_variance[1])
        x += np.cos(theta) * dist
        y += np.sin(theta) * dist

        self.position = np.array([x, y, theta]).astype(np.float32)


# class Sensor(object):
#     def __init__(self, landmarks, measurement_variance, max_range, dropout):
#         self.landmarks = landmarks
#         self.measurement_variance = measurement_variance
#         self.max_range = max_range
#         self.dropout = dropout

#     def __get_noisy_measurement(self, position, landmark, measurement_variance):
#         vector_to_landmark = np.array(landmark - position, dtype=np.float32)

#         a = np.random.normal(0, measurement_variance[0])
#         vector_to_landmark[0] += a
#         b = np.random.normal(0, measurement_variance[1])
#         vector_to_landmark[1] += b

#         return vector_to_landmark

#     def get_noisy_measurements(self, position):
#         visible_measurements = []
#         for i, landmark in enumerate(landmarks):
#             z = self.__get_noisy_measurement(position, landmark, measurement_variance)

#             coin_toss = np.random.uniform(0, 1)
#             if dist(landmark, position) < self.max_range and coin_toss < self.dropout:
#                 visible_measurements.append(z)

#         return np.array(visible_measurements, dtype=np.float32)


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
        particles, threads, block_size, z_real, observation_variance, cuda_particles,
        cuda_measurements, cuda_cov, threshold, max_range, max_fov):

    measurements = z_real.astype(np.float32)

    measurement_cov = np.float32([
        observation_variance[0], 0,
        0, observation_variance[1]
    ])

    start = cuda.Event()
    end = cuda.Event()
    start.record()

    cuda.memcpy_htod(cuda_particles, particles)
    cuda.memcpy_htod(cuda_measurements, measurements)
    cuda.memcpy_htod(cuda_cov, measurement_cov)

    func = cuda_update.get_function("update")
    func(cuda_particles, np.int32(block_size), cuda_measurements,
         np.int32(FlatParticle.len(particles)), np.int32(len(measurements)),
         cuda_cov, np.float32(threshold), np.float32(max_range), np.float32(max_fov),
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
    N = 256  # number of particles
    SIM_LENGTH = 50  # number of simulation steps
    MAX_RANGE = 4  # max range of sensor
    MAX_FOV = (6/5)*np.pi
    DT = 0.5
    MISS_PROB = 0.1  # probability landmark in range will be missed
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
        np.tile([0.13, 0.7], (SIM_LENGTH, 1))
    ))

    real_position_history = [start_position]
    predicted_position_history = [FlatParticle.get_mean_position(particles)]
    weights_history = [FlatParticle.neff(particles)]

    vehicle = Vehicle(start_position, movement_variance, dt=DT)
    sensor = Sensor(vehicle, landmarks, [], measurement_variance, MAX_RANGE, MAX_FOV, MISS_PROB, 0)

    cuda_particles = cuda.mem_alloc(4 * N * (6 + 7*MAX_LANDMARKS))
    cuda_measurements = cuda.mem_alloc(4 * 2 * MAX_MEASUREMENTS)
    cuda_cov = cuda.mem_alloc(4 * 4)

    # plt.pause(5)

    for i in range(u.shape[0]):
        print(i)

        if PLOT:
            plt.pause(0.05)
            ax[0].clear()
            ax[0].set_xlim([-5, 20])
            ax[0].set_ylim([-5, 20])
            plot_landmarks(ax[0], landmarks)
            plot_history(ax[0], real_position_history, color='green')
            plot_history(ax[0], predicted_position_history, color='orange')
            plot_particles_grey(ax[0], particles)

        vehicle.move_noisy(u[i])
        real_position_history.append(vehicle.position)

        FlatParticle.predict(particles, u[i], sigmas=movement_variance, dt=DT)

        measurements = sensor.get_noisy_measurements()
        visible_measurements = np.vstack((measurements["observed"], measurements["phantomSeen"]))
        # visible_measurements = sensor.get_noisy_measurements(vehicle.position[:2])

        particles = update(
            particles, THREADS, BLOCK_SIZE, visible_measurements, measurement_variance,
            cuda_particles, cuda_measurements, cuda_cov, THRESHOLD, MAX_RANGE, MAX_FOV
        )

        predicted_position_history.append(FlatParticle.get_mean_position(particles))

        if PLOT:
            ax[0].clear()
            ax[0].set_xlim([-5, 20])
            ax[0].set_ylim([-5, 20])
            plot_landmarks(ax[0], landmarks)
            plot_history(ax[0], real_position_history, color='green')
            plot_history(ax[0], predicted_position_history, color='orange')
            if(visible_measurements.size != 0):
                plot_connections(ax[0], vehicle.position, visible_measurements + vehicle.position[:2])
            plot_particles_weight(ax[0], particles)
            if(visible_measurements.size != 0):
                plot_measurement(ax[0], vehicle.position[:2], visible_measurements, color="red")

            ax[1].clear()
            ax[1].set_xlim([-5, 20])
            ax[1].set_ylim([-5, 20])
            best = np.argmax(FlatParticle.w(particles))
            plot_landmarks(ax[1], landmarks, color="green")
            plot_landmarks(ax[1], FlatParticle.get_landmarks(particles, best), color="orange")
            covariances = FlatParticle.get_covariances(particles, best)

            for i, landmark in enumerate(FlatParticle.get_landmarks(particles, best)):
                plot_confidence_ellipse(ax[1], landmark, covariances[i], n_std=3)

            plot_sensor_fov(ax[0], vehicle, MAX_RANGE, MAX_FOV)
            plot_sensor_fov(ax[1], vehicle, MAX_RANGE, MAX_FOV)


        if FlatParticle.neff(particles) < 0.6*N:
            print("resampling..")
            particles = FlatParticle.resample(particles)

        weights_history.append(FlatParticle.neff(particles))


    print("Mean CUDA compute time: ", np.mean(cuda_time) / 1000, ", stdev: ", np.std(cuda_time) / 1000)
    deviation = mean_path_deviation(real_position_history, predicted_position_history)
    print(f"Mean path deviation: {deviation}")