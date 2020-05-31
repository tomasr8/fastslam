import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
from filterpy.monte_carlo import systematic_resample

from plotting import plot_connections, plot_history, plot_landmarks, plot_measurement, plot_particles_weight, plot_particles_grey
from data_association import assign

class Particle(object):
    def __init__(self, x, y, theta, n_landmarks, w):
        self.x = x
        self.y = y
        self.theta = theta
        self.n_landmarks = n_landmarks
        self.w = w
        self.landmark_means = np.zeros((n_landmarks, 2), dtype=np.float)
        self.landmark_covariances = np.zeros((n_landmarks, 2, 2), dtype=np.float)

    def copy(self):
        p = Particle(self.x, self.y, self.theta, self.n_landmarks, self.w)
        p.landmark_means = np.copy(self.landmark_means)
        p.landmark_covariances = np.copy(self.landmark_covariances)

        return p

    def add_landmarks(self, means, measurement_cov):
        n_old = len(self.landmark_means)
        n_new = len(means)

        self.n_landmarks = n_old + n_new

        landmark_means = np.zeros((n_old + n_new, 2), dtype=np.float)
        landmark_means[:n_old, :] = self.landmark_means
        landmark_means[n_old:, :] = means
        self.landmark_means = landmark_means

        landmark_covariances = np.zeros((n_old + n_new, 2, 2), dtype=np.float)
        landmark_covariances[:n_old, :] = self.landmark_covariances
        landmark_covariances[n_old:, :] = measurement_cov
        self.landmark_covariances = landmark_covariances


def get_initial_particles(n, xlim, ylim, n_landmarks):
    particles = []

    for _ in range(n):
        x = np.random.uniform(xlim[0], xlim[1])
        y = np.random.uniform(ylim[0], ylim[1])
        theta = np.random.uniform(0, 2*math.pi)

        p = Particle(x, y, theta, n_landmarks, 1/n)
        particles.append(p)

    return particles



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


def update(particles, landmarks, z_real, observation_variance):
    for p in particles:
        pos = np.array([p.x, p.y], dtype=np.float)

        # M = len(landmarks)
        dist = get_dist_matrix(p.landmark_means, z_real + pos[:2], p.landmark_covariances, np.diag(observation_variance))
        assignment, _, _ = assign(dist)

        print(assignment)



        for i, _ in enumerate(landmarks):

            # print(assignment[i])
            # print(type(assignment[i]))

            z_predicted, H_predicted = get_measurement(pos, p.landmark_means[i])
            # residual = z_real[assignment[i]] - z_predicted
            residual = z_real[i] - z_predicted


            Q = (H_predicted @ p.landmark_covariances[i] @ H_predicted.T) + np.diag(observation_variance)

            K = p.landmark_covariances[i] @ H_predicted.T @ np.linalg.pinv(Q)

            p.landmark_means[i] = p.landmark_means[i] + (K @ residual)
            p.landmark_covariances[i] = (np.eye(2) - (K @ H_predicted)) @ p.landmark_covariances[i]

            # p.w *= scipy.stats.multivariate_normal.pdf(z_real[assignment[i]], mean=z_predicted, cov=Q, allow_singular=True)
            p.w *= scipy.stats.multivariate_normal.pdf(z_real[i], mean=z_predicted, cov=Q, allow_singular=True)

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


def neff(particles):
    weights = [p.w for p in particles]
    return 1. / np.sum(np.square(weights))


def get_mean(paricles):
    weights = np.array([p.w for p in paricles], dtype=np.float)
    xs = np.array([p.x for p in particles], dtype=np.float)
    ys = np.array([p.y for p in particles], dtype=np.float)
    thetas = np.array([p.theta for p in particles], dtype=np.float)

    return [
        np.average(xs, weights=weights),
        np.average(ys, weights=weights),
        np.average(thetas, weights=weights)
    ]


def dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)



def get_dist_matrix(landmarks, measurements, landmarks_cov, measurement_cov):
    M = len(landmarks)
    dist = np.zeros((M, M), dtype=np.float)

    # print("============================")
    # print(landmarks)
    # print(measurements)
    # print("lc", landmarks_cov)
    # print("mc", measurement_cov)

    for i in range(M):
        for j in range(M):
            cov = landmarks_cov[i] + measurement_cov
            dist[i, j] = -1 * scipy.stats.multivariate_normal.pdf(landmarks[i], mean=measurements[j], cov=cov, allow_singular=True)
            # print("->", i, j, cov, dist[i, j])
    
    return dist


if __name__ == "__main__":

    np.random.seed(1)

    fig, ax = plt.subplots()
    ax.set_xlim([0, 17])
    ax.set_ylim([0, 17])

    NL = 5
    landmarks = np.zeros((NL, 2), dtype=np.float)
    landmarks[:, 0] = np.random.uniform(2, 13, NL)
    landmarks[:, 1] = np.random.uniform(2, 13, NL)

    N = 50
    particles = get_initial_particles(N, (1.5, 2.5), (1.5, 2.5), len(landmarks))
    # print(len(particles))

    for p in particles:
        p.x = 2 + np.random.normal(0, 0.1)
        p.y = 2 + np.random.normal(0, 0.1)
        p.theta = 0

    ss = np.array([2, 2, 0], dtype=np.float)
    sigmas = [0.2, 0.2]

    z_real = []
    for landmark in landmarks:
        z = get_measurement_stochastic(ss[:2], landmark, sigmas)
        z_real.append(z)

    z_real = np.array(z_real)

    for p in particles:
        p.landmark_means = np.copy(z_real) + ss[:2]

        for i in range(len(landmarks)):
            p.landmark_covariances[i] = np.copy(np.diag(sigmas))
        # print(p.landmark_means)

    # s = [1, 1, 1]
    u = np.vstack((
        np.tile([0.0, 0.7], (13, 1)),
        np.tile([0.2, 0.7], (14, 1)),
        np.tile([0.0, 0.7], (7, 1)),
        np.tile([0.18, 0.7], (20, 1)),
        np.tile([0.0, 0.7], (14, 1))
    ))

    # s_history = [s]
    ss_history = [ss]
    slam_history = [get_mean(particles)]

    for i in range(68):
        print(i)

        plt.pause(0.01)

        ax.clear()
        ax.set_xlim([0, 17])
        ax.set_ylim([0, 17])
        plot_landmarks(ax, landmarks)
        plot_history(ax, ss_history, color='green')
        plot_history(ax, slam_history, color='orange')
        # plot_connection(ax, ss, z_real[landmark_indices, :] + ss[:2])
        plot_particles_grey(ax, particles)

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
            z = get_measurement_stochastic(ss[:2], landmark, sigmas)
            z_real.append(z)

        z_real = np.array(z_real)
        plot_measurement(ax, ss[:2], z_real, color="red")

        # print(z_real + ss[:2])

        update(particles, landmarks, z_real, sigmas)
        plt.pause(0.01)

        slam_history.append(get_mean(particles))

        ax.clear()
        ax.set_xlim([0, 17])
        ax.set_ylim([0, 17])
        plot_landmarks(ax, landmarks)
        plot_history(ax, ss_history, color='green')
        plot_history(ax, slam_history, color='orange')
        plot_connections(ax, ss, z_real + ss[:2])
        plot_particles_weight(ax, particles)
        plot_measurement(ax, ss[:2], z_real, color="red")


        if neff(particles) < N/2:
            print("resample", neff(particles))
            weights = [p.w for p in particles]
            indexes = systematic_resample(weights)
            # print(indexes)
            particles = resample_from_index(particles, indexes)
        # print(particles)
    # plt.show()