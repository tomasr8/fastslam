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


    @staticmethod
    def flatten_particles(particles, max_particles):
        n_particles = len(particles)
        size = n_particles * (6 + 6*max_particles)
        particle_array = np.zeros(shape=size, dtype=np.float32)

        for i, p in enumerate(particles):
            offset = i * (6 + 6*max_particles)
            particle_array[offset] = float(p.x)
            particle_array[offset+1] = float(p.y)
            particle_array[offset+2] = float(p.theta)
            particle_array[offset+3] = float(p.w)
            particle_array[offset+4] = float(max_particles)
            particle_array[offset+5] = float(p.n_landmarks)

            for j, mean in enumerate(p.landmark_means):
                particle_array[offset+6+2*j] = float(mean[0])
                particle_array[offset+6+2*j+1] = float(mean[1])

            for j, cov in enumerate(p.landmark_covariances):
                particle_array[offset+6+2*max_particles+4*j] = float(cov[0, 0])
                particle_array[offset+6+2*max_particles+4*j+1] = float(cov[0, 1])
                particle_array[offset+6+2*max_particles+4*j+2] = float(cov[1, 0])
                particle_array[offset+6+2*max_particles+4*j+3] = float(cov[1, 1])

        return particle_array

    @staticmethod
    def unflatten_particles(particles, particle_array, max_particles):
        for i, p in enumerate(particles):
            offset = i * (6 + 6*max_particles)
            p.w = particle_array[offset+3]
            p.n_landmarks = int(particle_array[offset+5])

            landmark_means = np.zeros((p.n_landmarks, 2), dtype=np.float32)
            landmark_covariances = np.zeros((p.n_landmarks, 2, 2), dtype=np.float32)

            for j in range(p.n_landmarks):
                landmark_means[j][0] = particle_array[offset+6+2*j]
                landmark_means[j][1] = particle_array[offset+6+2*j+1]

                landmark_covariances[j][0, 0] = particle_array[offset+6+2*max_particles+4*j]
                landmark_covariances[j][0, 1] = particle_array[offset+6+2*max_particles+4*j+1]
                landmark_covariances[j][1, 0] = particle_array[offset+6+2*max_particles+4*j+2]
                landmark_covariances[j][1, 1] = particle_array[offset+6+2*max_particles+4*j+3]

            p.landmark_means = landmark_means
            p.landmark_covariances = landmark_covariances


# class FlatParticle(object):
#     def __init__(self, x: float, y: float, theta: float, n_landmarks: int, w: float):
#         self.x = x
#         self.y = y
#         self.theta = theta
#         self.n_landmarks = n_landmarks
#         self.w = w
#         self.landmark_means = np.zeros((n_landmarks, 2), dtype=np.float)
#         self.landmark_covariances = np.zeros((n_landmarks, 2, 2), dtype=np.float)