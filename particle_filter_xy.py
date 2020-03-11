import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
from filterpy.monte_carlo import systematic_resample

def move_vehicle_exact(s, u, dt):
    x, y, theta = s
    omega, v = u

    theta += omega

    dist = (v * dt)
    x += np.cos(theta) * dist 
    y += np.sin(theta) * dist

    return np.array([x, y, theta])


def move_vehicle_stochastic(s, u, dt, sigmas):
    x, y, theta = s
    omega, v = u

    theta += omega + np.random.normal(0, sigmas[0])

    dist = (v * dt) + np.random.normal(0, sigmas[1])
    x += np.cos(theta) * dist 
    y += np.sin(theta) * dist

    return np.array([x, y, theta])


def get_landmark_measurements(s, landmarks, sigmas):
    n = len(landmarks)
    x, y, _ = s

    measurements = np.zeros((n, 2))
    for i, (a, b) in enumerate(landmarks):
        measurements[i] = [
            (a - x) + np.random.normal(0, sigmas[0]),
            (b - y) + np.random.normal(0, sigmas[1])
        ]

    return measurements


def predict(particles, u, sigmas, dt):
    N = len(particles)
    # update heading
    particles[:, 2] += u[0] + np.random.normal(0, sigmas[0], N)
    # particles[:, 2] %= 2 * np.pi

    # move in the (noisy) commanded direction
    dist = (u[1] * dt) + np.random.normal(0, sigmas[1], N)
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist

    return particles


def get_initial_particles(n=100, xlim=(0, 10), ylim=(0, 10)):
    x = np.random.uniform(xlim[0], xlim[1], n)
    y = np.random.uniform(ylim[0], ylim[1], n)
    theta = np.random.uniform(0, 2*math.pi, n)

    return np.array([x, y, theta]).T


def update(particles, weights, z, R, landmarks):
    for i, landmark in enumerate(landmarks):
        # distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        # weights *= scipy.stats.norm(distance, R).pdf(z[i])
        for j, p in enumerate(particles):
            vector = landmark - p[0:2]
            weights[j] *= scipy.stats.multivariate_normal.pdf(z[i], vector, R)

    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize

    return weights


def simple_resample(particles, weights):
    N = len(particles)
    print(N)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, np.random.normal(N))
    print(indexes)

    # resample according to indexes
    particles[:] = particles[indexes]
    weights.fill(1.0 / N)

    return particles, weights

# def plot_step(ax, s, color='green'):
#     x, y, theta = s
#     ax.plot([x], [y], marker='o', markersize=3, color=color)


def plot_history(ax, history, color='green'):
    for x, y, _ in history:
        ax.plot([x], [y], marker='o', markersize=3, color=color)  


def plot_landmarks(ax, landmarks, color='blue'):
    ax.scatter(landmarks[:, 0], landmarks[:, 1], marker='x', color=color)


def plot_connection(ax, s, landmarks, color='purple'):
    x, y, _ = s
    for a, b in landmarks:
        ax.plot([x, a], [y, b], color=color)


def plot_particles(ax, particles, color='grey'):
    ax.scatter(particles[:, 0], particles[:, 1], marker='o', color=color, s=2)


def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights.fill(1.0 / len(weights))

    return particles, weights


def neff(weights):
    return 1. / np.sum(np.square(weights))


if __name__ == "__main__":

    fig, ax = plt.subplots()
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 15])

    N = 2000
    particles = get_initial_particles(n=N, xlim=(0.5, 1.5), ylim=(0.5, 1.5))
    weights = np.ones(N) / N
    # print(particles)

    landmarks = np.array([
        [5, 2],
        [7, 6],
        [2, 8],
        [2, 3],
        [5, 5],
        [4, 10],
        [10, 3],
        [11, 6],
        [10, 9]
    ])

    s = [1, 1, 1]
    ss = [1, 1, 1]
    # u = [-0.05, 0.7]


    u = np.vstack((
        np.tile([-0.1, 0.7], (15, 1)),
        np.tile([0.2, 0.7], (25, 1))
    ))

    s_history = [s]
    ss_history = [ss]

    for i in range(35):
        plot_landmarks(ax, landmarks)
        plot_history(ax, s_history, color='green')
        plot_history(ax, ss_history, color='red')

        # plot_step(ax, s)
        # plot_step(ax, ss, color='red')
        plot_connection(ax, ss, landmarks)
        plot_particles(ax, particles)
        plt.pause(0.5)

        s = move_vehicle_exact(s, u[i], dt=1)
        ss = move_vehicle_stochastic(ss, u[i], dt=1, sigmas=[0.05, 0.2])
        s_history.append(s)
        ss_history.append(ss)

        particles = predict(particles, u[i], sigmas=[0.05, 0.2], dt=1)
        # print(particles)

        sigmas = [0.05, 0.05]
        R = np.array([
            [0.05, 0],
            [0, 0.05]
        ])

        weights = update(particles, weights, get_landmark_measurements(ss, landmarks, sigmas), R, landmarks)

        if neff(weights) < N/2:
            indexes = systematic_resample(weights)
            # print(indexes)
            particles, weights = resample_from_index(particles, weights, indexes)
        # print(particles)

        ax.clear()
        ax.set_xlim([0, 15])
        ax.set_ylim([0, 15])
    # plt.show()