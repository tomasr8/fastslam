import math
import numpy as np
from filterpy.monte_carlo import systematic_resample

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


class FlatParticle(object):
    @staticmethod
    def x(particles):
        max_landmarks = int(particles[4])
        step = 6 + 6*max_landmarks
        return particles[0::step]

    @staticmethod
    def y(particles):
        max_landmarks = int(particles[4])
        step = 6 + 6*max_landmarks
        return particles[1::step]

    @staticmethod
    def w(particles):
        max_landmarks = int(particles[4])
        step = 6 + 6*max_landmarks
        return particles[3::step]

    @staticmethod
    def len(particles):
        max_landmarks = int(particles[4])
        length = particles.shape[0]
        return int(length/(6 + 6*max_landmarks))

    @staticmethod
    def mean_particles(particles):
        max_landmarks = int(particles[4])
        step = 6 + 6*max_landmarks
        return np.mean(particles[5::step])

    @staticmethod
    def get_particle(particles, i):
        max_landmarks = int(particles[4])
        size = 6 + 6*max_landmarks
        offset = size * i
        return particles[offset:offset+size]

    @staticmethod
    def get_mean_position(particles) -> list:
        max_landmarks = int(particles[4])
        step = 6 + 6*max_landmarks

        xs = particles[0::step]
        ys = particles[1::step]
        thetas = particles[2::step]
        weights = particles[3::step]

        return [
            np.average(xs, weights=weights),
            np.average(ys, weights=weights),
            np.average(thetas, weights=weights)
        ]

    @staticmethod
    def get_initial_particles(n_particles: int, max_landmarks: int, starting_position: np.ndarray, sigma: float):
        step = 6 + 6*max_landmarks
        particles = np.zeros(n_particles * step, dtype=np.float32)

        particles[0::step] = starting_position[0] + np.random.normal(loc=0, scale=sigma, size=n_particles)
        particles[1::step] = starting_position[1] + np.random.normal(loc=0, scale=sigma, size=n_particles)
        particles[2::step] = starting_position[2] + (np.random.normal(loc=0, scale=sigma, size=n_particles) % (2*math.pi))
        particles[3::step] = 1/n_particles
        particles[4::step] = float(max_landmarks)

        return particles

    @staticmethod
    def predict(particles, u, dt, sigmas):
        '''Stochastically moves particles based on the control input and noise

        '''
        N = FlatParticle.len(particles)
        max_landmarks = int(particles[4])
        step = 6 + 6*max_landmarks

        particles[2::step] += u[0]
        # add noise to heading
        print("N", N, "step", step)
        particles[2::step] += np.random.normal(loc=0, scale=sigmas[0], size=N)

        # move in the (noisy) commanded direction
        theta = particles[2::step]
        dist = (u[1] * dt) + np.random.normal(loc=0, scale=sigmas[1], size=N)
        particles[0::step] += np.cos(theta) * dist
        particles[1::step] += np.sin(theta) * dist

    @staticmethod
    def neff(particles) -> float:
        weights = FlatParticle.w(particles)
        return 1.0/np.sum(np.square(weights))

    @staticmethod
    def resample_particles(particles):
        '''Resamples particles using systematic resample from filterpy

        '''
        N = FlatParticle.len(particles)
        max_landmarks = int(particles[4])
        size = 6 + 6*max_landmarks

        weights = FlatParticle.w(particles)
        indexes = systematic_resample(weights)

        new_particles = particles.copy()
        for i, index in enumerate(indexes):
            old_offset = size * index
            particle = particles[old_offset:old_offset+size].copy()
            particle[3] = 1.0/N

            new_offset = size * i
            new_particles[new_offset:new_offset+size] = particle

        return new_particles

    @staticmethod
    def rescale(particles):
        max_landmarks = int(particles[4])
        step = 6 + 6*max_landmarks

        s = np.sum(particles[3::step]) + 1.e-30
        particles[3::step] /= s