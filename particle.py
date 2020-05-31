import math
import numpy as np


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

    @staticmethod
    def get_initial_particles(n: int, xlim: float, ylim: float, n_landmarks: int):
        particles = []

        for _ in range(n):
            x = np.random.uniform(xlim[0], xlim[1])
            y = np.random.uniform(ylim[0], ylim[1])
            theta = np.random.uniform(0, 2*math.pi)

            p = Particle(x, y, theta, n_landmarks, 1/n)
            particles.append(p)

        return particles

    @staticmethod
    def get_mean_position(particles):
        weights = np.array([p.w for p in particles], dtype=np.float)
        xs = np.array([p.x for p in particles], dtype=np.float)
        ys = np.array([p.y for p in particles], dtype=np.float)
        thetas = np.array([p.theta for p in particles], dtype=np.float)

        return [
            np.average(xs, weights=weights),
            np.average(ys, weights=weights),
            np.average(thetas, weights=weights)
        ]
