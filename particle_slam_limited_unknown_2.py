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
from data_association import associate_landmarks_measurements
from utils import dist

import numba as nb
from multiprocessing import Process

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

from cuda.update import cuda_update



def get_measurement(position, landmark):
    '''
        Returns the ideal (without noise) measurement and the Jacobian
    '''
    vector_to_landmark = np.array(landmark - position, dtype=np.float)

    jacobian = np.array([
        [1, 0],
        [0, 1]
    ], dtype=np.float)

    return vector_to_landmark, jacobian


def get_measurement_stochastic(position, landmark, measurement_variance):
    '''
        Returns a noisy measurement given by the variance
    '''
    vector_to_landmark = np.array(landmark - position, dtype=np.float)

    a = np.random.normal(0, measurement_variance[0])
    vector_to_landmark[0] += a
    b = np.random.normal(0, measurement_variance[1])
    vector_to_landmark[1] += b

    return vector_to_landmark


def move_vehicle_stochastic(pos, u, dt, sigmas):
    '''
        Stochastically moves the vehicle based on the control input and noise
    '''
    x, y, theta = pos

    theta += u[0] + np.random.normal(0, sigmas[0])

    dist = (u[1] * dt) + np.random.normal(0, sigmas[1])
    x += np.cos(theta) * dist
    y += np.sin(theta) * dist

    return np.array([x, y, theta], dtype=np.float)


def update(particles, z_real, observation_variance, cuda_particles, cuda_measurements, cuda_cov):
    measurements = z_real.astype(np.float32)

    measurement_cov = np.float32([
        observation_variance[0], 0,
        0, observation_variance[1]
    ])

    #Copies the memory from CPU to GPU
    # start = time.time()
    cuda.memcpy_htod(cuda_particles, particles)
    cuda.memcpy_htod(cuda_measurements, measurements)
    cuda.memcpy_htod(cuda_cov, measurement_cov)

    threshold = 0.1

    func = cuda_update.get_function("update")
    func(cuda_particles, cuda_measurements, np.int32(FlatParticle.len(particles)), np.int32(len(measurements)), cuda_cov, np.float32(threshold), block=(64, 1, 1), grid=(32, 1, 1))

    # new_particles = np.empty_like(particles)
    cuda.memcpy_dtoh(particles, cuda_particles)
    # print("cuda_took: ", (time.time() - start))

    # start = time.time()
    FlatParticle.rescale(particles)

    # print("rescale took:", time.time() - start)

    return particles


if __name__ == "__main__":
    np.random.seed(2)

    PLOT = True
    MAX_DIST = 3

    if PLOT:
        fig, ax = plt.subplots()
        ax.set_xlim([0, 15])
        ax.set_ylim([0, 15])

    # NL = 6
    # landmarks = np.zeros((NL, 2), dtype=np.float)
    # landmarks[:, 0] = np.random.uniform(2, 10, NL)
    # landmarks[:, 1] = np.random.uniform(2, 10, NL)

    landmarks = np.array([
        [9, 2],
        [11, 3],
        [13, 5],
        [13, 6],
        [13, 7],
        [13, 8],
        [13, 9],
        [12, 12],
        [10, 13],
        [8, 13],
        [6, 12],
        [4, 12],
        [3, 11],
        [3, 10],
        [2, 10],
        [2, 9],
        [2, 7],
        [2, 5],
        [3, 3],
        [3, 2],
        [4, 2],
        [5, 2],
        [6, 2],
        [6, 3],
        [7, 3],
    ], dtype=np.float)
    NL = landmarks.shape[0]
    
    real_position = np.array([8, 3, 0], dtype=np.float)

    N = 2048
    MAX_LANDMARKS = 100
    particles = FlatParticle.get_initial_particles(N, MAX_LANDMARKS, real_position, sigma=0.2)

    u = np.vstack((
        np.tile([0.13, 0.7], (120, 1)),
        # np.tile([0.3, 0.7], (4, 1)),
        # np.tile([0.0, 0.7], (6, 1)),
        # np.tile([0.3, 0.7], (5, 1)),
        # np.tile([0.0, 0.7], (11, 1)),
        # np.tile([0.3, 0.7], (4, 1)),
    ))

    real_position_history = [real_position]
    predicted_position_history = [FlatParticle.get_mean_position(particles)]

    movement_variance = [0.03, 0.05]
    measurement_variance = [0.1, 0.1]

    # print(particles.nbytes / N)
    # raise Exception

    cuda_particles = cuda.mem_alloc(2424 * N)
    cuda_measurements = cuda.mem_alloc(128)
    cuda_cov = cuda.mem_alloc(16)

    for i in range(u.shape[0]):
        loop_time = time.time()
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


        real_position = move_vehicle_stochastic(real_position, u[i], dt=1, sigmas=movement_variance)
        real_position_history.append(real_position)

        FlatParticle.predict(particles, u[i], sigmas=movement_variance, dt=1)

        z_real = []
        visible_landmarks = []
        landmark_indices = []
        for i, landmark in enumerate(landmarks):
            z = get_measurement_stochastic(real_position[:2], landmark, measurement_variance)
            z_real.append(z)

            if dist(landmark, real_position) < MAX_DIST:
                visible_landmarks.append(landmark)
                landmark_indices.append(i)

        z_real = np.array(z_real)
        visible_landmarks = np.array(visible_landmarks, dtype=np.float)
        # plot_measurement(ax, real_position[:2], z_real, color="red")

        # start = time.time()
        particles = update(particles, z_real[landmark_indices], measurement_variance, cuda_particles, cuda_measurements, cuda_cov)
        # print(particles.nbytes)
        # print("took: ", (time.time() - start))
        # plt.pause(2)

        predicted_position_history.append(FlatParticle.get_mean_position(particles))

        if PLOT:
            ax.clear()
            ax.set_xlim([0, 15])
            ax.set_ylim([0, 15])
            plot_landmarks(ax, landmarks)
            plot_history(ax, real_position_history, color='green')
            plot_history(ax, predicted_position_history, color='orange')
            plot_connections(ax, real_position, z_real[landmark_indices, :] + real_position[:2])
            plot_particles_weight(ax, particles)
            plot_measurement(ax, real_position[:2], z_real, color="red")

        if FlatParticle.neff(particles) < N/2:
            print("resample", FlatParticle.neff(particles))
            particles = FlatParticle.resample_particles(particles)

        print("loop time:", time.time() - loop_time)