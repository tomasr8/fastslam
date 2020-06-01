import math
import numpy as np

class Particle(object):
    def __init__(self, x: float, y: float, theta: float, n_landmarks: int, w: float):
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
    def get_initial_particles(n_particles: int, starting_position: np.ndarray, sigma: float = 0.2):
        particles = [None] * n_particles
        n_landmarks = 0

        for i in range(n_particles):
            x = starting_position[0] + np.random.normal(0, sigma)
            y = starting_position[1] + np.random.normal(0, sigma)
            theta = starting_position[2] = np.random.normal(0, sigma) % (2*math.pi)

            weight = 1/n_particles

            p = Particle(x, y, theta, n_landmarks, weight)
            particles[i] = p

        return particles

    @staticmethod
    def get_mean_position(particles) -> list:
        weights = np.array([p.w for p in particles], dtype=np.float)
        xs = np.array([p.x for p in particles], dtype=np.float)
        ys = np.array([p.y for p in particles], dtype=np.float)
        thetas = np.array([p.theta for p in particles], dtype=np.float)

        return [
            np.average(xs, weights=weights),
            np.average(ys, weights=weights),
            np.average(thetas, weights=weights)
        ]
