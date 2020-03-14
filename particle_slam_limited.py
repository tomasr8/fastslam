import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
from filterpy.monte_carlo import systematic_resample


class Particle(object):
    def __init__(self, x, y, theta, n_landmarks, w):
        self.x = x
        self.y = y
        self.theta = theta
        self.n_landmarks = n_landmarks
        self.w = w
        self.landmark_means = np.zeros((n_landmarks, 2))
        self.landmark_covariances = np.zeros((n_landmarks, 2, 2))

    def copy(self):
        p = Particle(self.x, self.y, self.theta, self.n_landmarks, self.w)
        p.landmark_means = np.copy(self.landmark_means)
        p.landmark_covariances = np.copy(self.landmark_covariances)

        return p


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


def update(particles, landmark_indices, landmarks, z_real, observation_variance):
    for i in landmark_indices:
        landmark = landmarks[i]

        for p in particles:
            pos = np.array([p.x, p.y], dtype=np.float)

            # print("pos", pos)
            # print(p.landmark_means[i])

            z_predicted, H_predicted = get_measurement(pos, p.landmark_means[i])
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

    if len(history) > 1:
        for i in range(len(history) - 1):
            x, y, _ = history[i]
            a, b, _ = history[i + 1]
            ax.plot([x, a], [y, b], color=color)
     

def plot_slam(ax, history, color='green'):
    for x, y in history:
        ax.plot([x], [y], marker='o', markersize=3, color=color)

    if len(history) > 1:
        for i in range(len(history) - 1):
            x, y = history[i]
            a, b = history[i + 1]
            ax.plot([x, a], [y, b], color=color)


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


def dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)



if __name__ == "__main__":

    np.random.seed(1)

    MAX_DIST = 1.3

    fig, ax = plt.subplots()
    ax.set_xlim([0, 17])
    ax.set_ylim([0, 17])

    NL = 80
    landmarks = np.zeros((NL, 2), dtype=np.float)
    landmarks[:, 0] = np.random.uniform(2, 13, NL)
    landmarks[:, 1] = np.random.uniform(2, 13, NL)

    N = 1000
    particles = get_initial_particles(N, (1.5, 2.5), (1.5, 2.5), len(landmarks))
    # print(len(particles))

    for p in particles:
        p.x = 2 + np.random.normal(0, 0.1)
        p.y = 2 + np.random.normal(0, 0.1)
        p.theta = 0

    ss = np.array([2, 2, 0], dtype=np.float)
    sigmas = [0.2, 0.2]

    z_real = []
    visible_landmarks = []
    landmark_indices = []
    for i, landmark in enumerate(landmarks):
        z = get_measurement_stochastic(ss[:2], landmark, sigmas)
        z_real.append(z)

        if dist(landmark, ss) < MAX_DIST:
            visible_landmarks.append(landmark)
            landmark_indices.append(i)

    z_real = np.array(z_real, dtype=np.float)
    visible_landmarks = np.array(visible_landmarks, dtype=np.float)

    print("z_real", z_real[landmark_indices, :])

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
        plot_slam(ax, slam_history, color='orange')
        # plot_connection(ax, ss, z_real[landmark_indices, :] + ss[:2])
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
        visible_landmarks = []
        landmark_indices = []
        for i, landmark in enumerate(landmarks):
            z = get_measurement_stochastic(ss[:2], landmark, sigmas)
            z_real.append(z)

            if dist(landmark, ss) < MAX_DIST:
                visible_landmarks.append(landmark)
                landmark_indices.append(i)

        z_real = np.array(z_real)
        visible_landmarks = np.array(visible_landmarks, dtype=np.float)
        plot_measurement(ax, ss[:2], z_real, color="red")

        # print(z_real + ss[:2])

        update(particles, landmark_indices, landmarks, z_real, sigmas)
        plt.pause(0.01)

        slam_history.append(get_mean(particles))

        ax.clear()
        ax.set_xlim([0, 17])
        ax.set_ylim([0, 17])
        plot_landmarks(ax, landmarks)
        plot_history(ax, ss_history, color='green')
        plot_slam(ax, slam_history, color='orange')
        plot_connection(ax, ss, z_real[landmark_indices, :] + ss[:2])
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