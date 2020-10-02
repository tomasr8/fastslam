import math
import time
from typing import List
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
from filterpy.monte_carlo import systematic_resample

from plotting import plot_connections, plot_history, plot_landmarks, plot_measurement, plot_particles_weight, plot_particles_grey
from particle import Particle
from data_association import associate_landmarks_measurements
from utils import dist, neff

import numba as nb



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


def predict(particles, u, dt, sigmas):
    '''
        Stochastically moves particles based on the control input and noise
    '''
    N = len(particles)

    for p in particles:
        # add noise to heading
        p.theta += u[0] + np.random.normal(0, sigmas[0])

        # move in the (noisy) commanded direction
        dist = (u[1] * dt) + np.random.normal(0, sigmas[1])
        p.x += np.cos(p.theta) * dist
        p.y += np.sin(p.theta) * dist

    return particles


def pdf_2(x, mean, cov):
    a, b = cov[0, :]

    logdet = math.log(a*a - b*b)

    root = math.sqrt(2)/2
    e = root * (a-b)**(-0.5)
    f = root * (a+b)**(-0.5)

    m = x[0] - mean[0]
    n = x[1] - mean[1]

    maha = 2*(m*m*e*e + n*n*f*f)
    log2pi = math.log(2 * math.pi)
    return math.exp(-0.5 * (2*log2pi + maha + logdet))


def pinv_2(A):
    a = A[0, 0]
    b = A[0, 1]
    c = A[1, 0]
    d = A[1, 1]

    e = a*a + c*c
    f = a*b + c*d
    g = a*b + c*d
    h = b*b + d*d

    scalar = 1/(e*h - f*g)
    e_i = scalar * h
    f_i = scalar * (-f)
    g_i = scalar * (-g)
    h_i = scalar * e

    return np.array([
        [e_i*a + f_i*b, e_i*c + f_i*d],
        [g_i*a + h_i*b, g_i*c + h_i*d]
    ], dtype=np.float)


def update(particles, z_real, observation_variance):
    # z_real = z_real[landmark_indices]

    for p in particles:
        pos = np.array([p.x, p.y], dtype=np.float)

        assignment, unassigned_measurement_idx = associate_landmarks_measurements(
            p, z_real, np.diag(observation_variance), threshold=0.1
        )

        # print(assignment)

        unassigned_measurement_idx = np.array(list(unassigned_measurement_idx), dtype=int)
        # print(unassigned_measurement_idx)
        unassigned = z_real[unassigned_measurement_idx]
        p.add_landmarks(unassigned + np.array([p.x, p.y]), np.diag(observation_variance))

        for i in assignment:
            j = assignment[i]

            z_predicted, H_predicted = get_measurement(pos, p.landmark_means[i])
            residual = z_real[j] - z_predicted

            # print("residual", residual)

            Q = (H_predicted @ p.landmark_covariances[i] @ H_predicted.T) + np.diag(observation_variance)

            # print("Q", Q)

            K = p.landmark_covariances[i] @ H_predicted.T @ pinv_2(Q)
            # K = p.landmark_covariances[i] @ H_predicted.T @ np.linalg.pinv(Q)

            # print("K", K)
            # print("K_res", K @ residual)
            p.landmark_means[i] = p.landmark_means[i] + (K @ residual)
            p.landmark_covariances[i] = (np.eye(2) - (K @ H_predicted)) @ p.landmark_covariances[i]

            p.w *= pdf_2(z_real[j], mean=z_predicted, cov=Q)
            # p.w *= scipy.stats.multivariate_normal.pdf(z_real[j], mean=z_predicted, cov=Q, allow_singular=False)


    s = 0
    for p in particles:
        p.w += 1.e-300
        s += p.w

    for p in particles:
        p.w /= s


def resample_from_index(particles: List[Particle], indexes) -> List[Particle]:
    '''

    '''
    N = len(particles)
    new_particles = []

    for i in indexes:
        p = particles[i].copy()
        p.w = 1 / N
        new_particles.append(p)

    return new_particles


def resample_particles(particles: List[Particle]) -> List[Particle]:
    '''
        Resamples particles using systematic resample from filterpy
    '''
    N = len(particles)
    weights = [p.w for p in particles]
    indexes = systematic_resample(weights)

    new_particles = []
    for i in indexes:
        p = particles[i].copy()
        p.w = 1 / N
        new_particles.append(p)

    return new_particles


if __name__ == "__main__":
    np.random.seed(2)

    PLOT = False
    MAX_DIST = 3

    fig, ax = plt.subplots()
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 15])

    # NL = 14
    # landmarks = np.array([
    #     [3, 1],
    #     [4, 3],
    #     [5, 1],
    #     [6, 3],
    #     [7, 1],
    #     [8, 3],
    #     [9, 1],
    #     [10, 3],
    #     [11, 1],
    #     [12, 3]
    # ], dtype=np.float)
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

    N = 1024
    particles = Particle.get_initial_particles(N, real_position, sigma=0.2)

    u = np.vstack((
        np.tile([0.13, 0.7], (120, 1)),
        # np.tile([0.3, 0.7], (4, 1)),
        # np.tile([0.0, 0.7], (6, 1)),
        # np.tile([0.3, 0.7], (5, 1)),
        # np.tile([0.0, 0.7], (11, 1)),
        # np.tile([0.3, 0.7], (4, 1)),
    ))

    real_position_history = [real_position]
    predicted_position_history = [Particle.get_mean_position(particles)]

    movement_variance = [0.03, 0.05]
    measurement_variance = [0.1, 0.1]

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

        real_position = move_vehicle_stochastic(real_position, u[i], dt=1, sigmas=movement_variance)
        real_position_history.append(real_position)

        particles = predict(particles, u[i], sigmas=movement_variance, dt=1)

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

        start = time.time()
        update(particles, z_real[landmark_indices], measurement_variance)
        print("took: ", (time.time() - start))
        # plt.pause(2)

        predicted_position_history.append(Particle.get_mean_position(particles))

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

        if neff(particles) < N/2:
            print("resample", neff(particles))
            particles = resample_particles(particles)
