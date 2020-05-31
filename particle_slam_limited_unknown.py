import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
from filterpy.monte_carlo import systematic_resample

from plotting import plot_connections, plot_history, plot_landmarks, plot_measurement, plot_particles_weight, plot_particles_grey
from particle import Particle
from data_association import associate_landmarks_measurements
from utils import dist, neff

def get_measurement_stochastic(position, landmark, observation_variance):
    vector_to_landmark = np.array(landmark - position, dtype=np.float)

    a = np.random.normal(0, observation_variance[0])
    vector_to_landmark[0] += a
    b = np.random.normal(0, observation_variance[1])
    vector_to_landmark[1] += b

    return vector_to_landmark


def get_measurement(position, landmark):
    vector_to_landmark = np.array(landmark - position, dtype=np.float)

    jacobian = np.array([
        [1, 0],
        [0, 1]
    ], dtype=np.float)

    return vector_to_landmark, jacobian



def move_vehicle_stochastic(pos, u, dt, sigmas):
    x, y, theta = pos
    omega, v = u

    theta += omega + np.random.normal(0, sigmas[0])

    dist = (v * dt) + np.random.normal(0, sigmas[1])
    x += np.cos(theta) * dist 
    y += np.sin(theta) * dist

    return np.array([x, y, theta], dtype=np.float)


def predict(particles, u, dt, sigmas):
    N = len(particles)
    # update heading

    for p in particles:
        p.theta += u[0] + np.random.normal(0, sigmas[0])
        # particles[:, 2] %= 2 * np.pi

        # move in the (noisy) commanded direction
        dist = (u[1] * dt) + np.random.normal(0, sigmas[1])
        p.x += np.cos(p.theta) * dist
        p.y += np.sin(p.theta) * dist

    return particles


def update(particles, landmark_indices, z_real, observation_variance):
    z_real = z_real[landmark_indices]

    for p in particles:
        pos = np.array([p.x, p.y], dtype=np.float)

        assignment, unassigned_measurement_idx = associate_landmarks_measurements(
            p, z_real, observation_variance, 0.1
        )

        unassigned_measurement_idx = np.array(list(unassigned_measurement_idx), dtype=int)
        # print(unassigned_measurement_idx)
        unassigned = z_real[unassigned_measurement_idx]
        p.add_landmarks(unassigned + np.array([p.x, p.y]), observation_variance)

        for i in assignment:
            j = assignment[i]

            z_predicted, H_predicted = get_measurement(pos, p.landmark_means[i])
            residual = z_real[j] - z_predicted

            Q = (H_predicted @ p.landmark_covariances[i] @ H_predicted.T) + np.diag(observation_variance)

            K = p.landmark_covariances[i] @ H_predicted.T @ np.linalg.pinv(Q)

            p.landmark_means[i] = p.landmark_means[i] + (K @ residual)
            p.landmark_covariances[i] = (np.eye(2) - (K @ H_predicted)) @ p.landmark_covariances[i]

            p.w *= scipy.stats.multivariate_normal.pdf(z_real[j], mean=z_predicted, cov=Q, allow_singular=True)

    s = 0
    for p in particles:
        p.w += 1.e-300
        s += p.w

    for p in particles:
        p.w /= s


def resample_from_index(particles, indexes):
    N = len(particles)
    new_particles = []

    for i in indexes:
        p = particles[i].copy()
        p.w = 1 / N
        new_particles.append(p)

    return new_particles


if __name__ == "__main__":
    np.random.seed(2)

    MAX_DIST = 3

    fig, ax = plt.subplots()
    ax.set_xlim([0, 17])
    ax.set_ylim([0, 17])

    NL = 8
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
    landmarks = np.zeros((NL, 2), dtype=np.float)
    landmarks[:, 0] = np.random.uniform(2, 10, NL)
    landmarks[:, 1] = np.random.uniform(2, 10, NL)

    N = 500
    particles = Particle.get_initial_particles(N, (1.5, 3.5), (1.5, 3.5), 0)

    for p in particles:
        p.x = 2 + np.random.normal(0, 0.1)
        p.y = 2 + np.random.normal(0, 0.1)
        p.theta = 0

    real_position = np.array([2, 2, 0], dtype=np.float)

    u = np.vstack((
        np.tile([0.0, 0.7], (4, 1)),
        np.tile([0.3, 0.7], (31, 1)),
    ))

    real_position_history = [real_position]
    predicted_position_history = [Particle.get_mean_position(particles)]

    for i in range(35):
        print(i)

        plt.pause(0.01)

        ax.clear()
        ax.set_xlim([0, 17])
        ax.set_ylim([0, 17])
        plot_landmarks(ax, landmarks)
        plot_history(ax, real_position_history, color='green')
        plot_history(ax, predicted_position_history, color='orange')
        # plot_connection(ax, real_position, z_real[landmark_indices, :] + real_position[:2])
        plot_particles_grey(ax, particles)

        real_position = move_vehicle_stochastic(real_position, u[i], dt=1, sigmas=[0.1, 0.05])
        real_position_history.append(real_position)

        particles = predict(particles, u[i], sigmas=[0.1, 0.05], dt=1)

        sigmas = [0.1, 0.1]
        R = np.array([
            [sigmas[0], 0],
            [0, sigmas[1]]
        ], dtype=np.float)

        z_real = []
        visible_landmarks = []
        landmark_indices = []
        for i, landmark in enumerate(landmarks):
            z = get_measurement_stochastic(real_position[:2], landmark, sigmas)
            z_real.append(z)

            if dist(landmark, real_position) < MAX_DIST:
                visible_landmarks.append(landmark)
                landmark_indices.append(i)

        z_real = np.array(z_real)
        visible_landmarks = np.array(visible_landmarks, dtype=np.float)
        plot_measurement(ax, real_position[:2], z_real, color="red")


        update(particles, landmark_indices, z_real, sigmas)
        plt.pause(0.01)

        predicted_position_history.append(Particle.get_mean_position(particles))

        ax.clear()
        ax.set_xlim([0, 17])
        ax.set_ylim([0, 17])
        plot_landmarks(ax, landmarks)
        plot_history(ax, real_position_history, color='green')
        plot_history(ax, predicted_position_history, color='orange')
        plot_connections(ax, real_position, z_real[landmark_indices, :] + real_position[:2])
        plot_particles_weight(ax, particles)
        plot_measurement(ax, real_position[:2], z_real, color="red")


        if neff(particles) < N/2:
            print("resample", neff(particles))
            weights = [p.w for p in particles]
            indexes = systematic_resample(weights)
            # print(indexes)
            particles = resample_from_index(particles, indexes)