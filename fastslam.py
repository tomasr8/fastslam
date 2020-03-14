import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
from filterpy.monte_carlo import systematic_resample


def get_initial_particles(n, xlim, ylim, n_landmarks):
    particles = np.zeros((n, 3), dtype=np.float)

    particles[:, 0] = np.random.uniform(xlim[0], xlim[1], size=n)
    particles[:, 1] = np.random.uniform(ylim[0], ylim[1], size=n)
    particles[:, 2] = np.random.uniform(0, 2*math.pi, size=n)

    particles_landmark_means = np.zeros((n, n_landmarks, 2), dtype=np.float)
    particles_landmark_covariances = np.zeros((n, n_landmarks, 2, 2), dtype=np.float)

    particle_weights = np.ones(n, dtype=np.float) / n

    return particles, particles_landmark_means, particles_landmark_covariances, particle_weights



def get_measurement(position, landmark, observation_variance = 0):
    vector_to_landmark = np.array(landmark - position, dtype=np.float)

    if observation_variance != 0:
        a = np.random.normal(0, observation_variance[0])
        vector_to_landmark[0] += a
        b = np.random.normal(0, observation_variance[1])
        vector_to_landmark[1] += b

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
    n = particles.shape[0]

    particles[:, 2] = u[0] + np.random.normal(0, sigmas[0], size=n)
    
    dist = (u[1] * dt) + np.random.normal(0, sigmas[1], size=n)
    particles[:, 0] +=  np.cos(particles[:, 2]) * dist
    particles[:, 1] +=  np.sin(particles[:, 2]) * dist

    return particles


def update(particles, landmarks, z_real, observation_variance):
    for i, landmark in enumerate(landmarks):


        for p in particles:
            pos = np.array([p.x, p.y], dtype=np.float)

            # print("pos", pos)
            # print(p.landmark_means[i])

            z_predicted, H_predicted = get_measurement(pos, p.landmark_means[i], 0)
            # print(pos, z_real[i], p.landmark_means[i], z_predicted)
            residual = z_real[i] - z_predicted

            Q = (H_predicted @ p.landmark_covariances[i] @ H_predicted.T) + np.diag(observation_variance)
            # print(p.landmark_covariances[i].shape)
            # print(H_predicted.T.shape)
            # print(np.linalg.pinv(Q).shape)

            # print(p.landmark_covariances[i], H_predicted.T @ np.linalg.pinv(Q))

            K = p.landmark_covariances[i] @ H_predicted.T @ np.linalg.pinv(Q)

            # print(K)
            # print(residual)

            p.landmark_means[i] = p.landmark_means[i] + (K @ residual)
            p.landmark_covariances[i] = (np.eye(2) - (K @ H_predicted)) @ p.landmark_covariances[i]

            p.w *= scipy.stats.multivariate_normal.pdf(z_real[i], mean=z_predicted, cov=Q, allow_singular=True)

    s = 0
    for p in particles:
        p.w += 1.e-300
        s += p.w

    for p in particles:
        p.w /= s

    # weights = [p.w for p in particles]
    # np.save("weights.npy", weights)
    # print(max(weights), min(weights))
    # plt.hist(weights)
    # plt.show()

    # return weights

def plot_history(ax, history, color='green'):
    for x, y, _ in history:
        ax.plot([x], [y], marker='o', markersize=3, color=color)  

def plot_slam(ax, history, color='green'):
    for x, y in history:
        ax.plot([x], [y], marker='o', markersize=3, color=color)  


def plot_landmarks(ax, landmarks, color='blue'):
    ax.scatter(landmarks[:, 0], landmarks[:, 1], marker='x', color=color)


def plot_measurement(ax, pos, landmarks, color):
    landmarks = landmarks + pos
    ax.scatter(landmarks[:, 0], landmarks[:, 1], s=20, marker='o', color=color)


def plot_connection(ax, s, landmarks, color='purple'):
    x, y, _ = s
    for a, b in landmarks:
        ax.plot([x, a], [y, b], color=color)


def plot_particles_c(ax, particles):
    pos = [[p.x, p.y] for p in particles]
    pos = np.array(pos, dtype=np.float)

    weight = [p.w for p in particles]

    ax.scatter(pos[:, 0], pos[:, 1], marker='o', c=weight, s=2)


def plot_particles(ax, particles, color='grey'):
    pos = [[p.x, p.y] for p in particles]
    pos = np.array(pos, dtype=np.float)

    ax.scatter(pos[:, 0], pos[:, 1], marker='o', color=color, s=2)


def resample_from_index(particles, indexes):
    N = len(particles)
    new_particles = []

    for i in indexes:
        p = particles[i].copy()
        p.w = 1 / N
        new_particles.append(p)

    return new_particles


def neff(particles):
    weights = [p.w for p in particles]
    return 1. / np.sum(np.square(weights))


def get_mean(paricles):
    weights = np.array([p.w for p in paricles], dtype=np.float)
    xs = np.array([p.x for p in particles], dtype=np.float)
    ys = np.array([p.y for p in particles], dtype=np.float)

    return [
        np.average(xs, weights=weights),
        np.average(ys, weights=weights)
    ]



if __name__ == "__main__":

    np.random.seed(1)

    fig, ax = plt.subplots()
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 15])

    MAX_DIST = 5

    landmarks = np.array([
        [5, 2],
        [7, 6],
        [2, 8],
        [2, 3],
        [5, 5]
    ], dtype=np.float)

    N = 1000
    particles = get_initial_particles(N, (0.5, 1.5), (0.5, 1.5), 5)
    # print(len(particles))

    for p in particles:
        p.x = 1 + np.random.normal(0, 0.1)
        p.y = 1 + np.random.normal(0, 0.1)
        p.theta = 1

    ss = np.array([1, 1, 1], dtype=np.float)
    sigmas = [0.1, 0.1]

    z_real = []
    for landmark in landmarks:
        z, _ = get_measurement(ss[:2], landmark, sigmas)
        z_real.append(z)
    z_real = np.array(z_real, dtype=np.float)

    print("z_real", z_real)

    for p in particles:
        p.landmark_means = np.copy(z_real) + ss[:2]

        for i in range(len(landmarks)):
            p.landmark_covariances[i] = np.copy(np.diag(sigmas))
        # print(p.landmark_means)

    # s = [1, 1, 1]
    u = np.vstack((
        np.tile([-0.09, 0.7], (15, 1)),
        np.tile([0.21, 0.7], (15, 1)),
        np.tile([0.07, 0.7], (15, 1))
    ))

    # s_history = [s]
    ss_history = [ss]
    slam_history = [get_mean(particles)]

    for i in range(45):
        print(i)

        plt.pause(0.5)

        ax.clear()
        ax.set_xlim([0, 15])
        ax.set_ylim([0, 15])
        plot_landmarks(ax, landmarks)
        plot_history(ax, ss_history, color='green')
        plot_slam(ax, slam_history, color='orange')
        # plot_connection(ax, ss, landmarks)
        plot_particles(ax, particles)

        # s = move_vehicle_exact(s, u[i], dt=1)
        ss = move_vehicle_stochastic(ss, u[i], dt=1, sigmas=[0.05, 0.2])
        # s_history.append(s)
        ss_history.append(ss)


        particles = predict(particles, u[i], sigmas=[0.05, 0.2], dt=1)
        # print(particles)

        # sigmas = [0.05, 0.05]
        R = np.array([
            [sigmas[0], 0],
            [0, sigmas[1]]
        ], dtype=np.float)

        # print(ss)

        z_real = []
        for landmark in landmarks:
            z, _ = get_measurement(ss[:2], landmark, sigmas)
            z_real.append(z)

        z_real = np.array(z_real)
        plot_measurement(ax, ss[:2], z_real, color="red")

        # print(z_real + ss[:2])

        update(particles, landmarks, z_real, sigmas)
        plt.pause(0.5)

        slam_history.append(get_mean(particles))

        ax.clear()
        ax.set_xlim([0, 15])
        ax.set_ylim([0, 15])
        plot_landmarks(ax, landmarks)
        plot_history(ax, ss_history, color='green')
        plot_slam(ax, slam_history, color='orange')
        # plot_connection(ax, ss, landmarks)
        plot_particles_c(ax, particles)
        plot_measurement(ax, ss[:2], z_real, color="red")


        if neff(particles) < N/2:
            print("resample", neff(particles))
            weights = [p.w for p in particles]
            indexes = systematic_resample(weights)
            # print(indexes)
            particles = resample_from_index(particles, indexes)
        # print(particles)
    # plt.show()