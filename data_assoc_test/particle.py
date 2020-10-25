import math
import numpy as np
from filterpy.monte_carlo import systematic_resample

class FlatParticle(object):
    @staticmethod
    def set_lm(particles, lm, cov):
        for i in range(FlatParticle.len(particles)):
            p = FlatParticle.get_particle(particles, i)
            p[5] = lm.shape[0]
            
            for j in range(lm.shape[0]):
                p[6+2*j:6+2*j+2] = lm[j]

                offset = 6 + lm.shape[0] * 2

                p[offset+4*j] = cov[0, 0]
                p[offset+4*j+1] = cov[0, 1]
                p[offset+4*j+2] = cov[1, 0]
                p[offset+4*j+3] = cov[1, 1]


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
    def get_initial_particles(n_particles: int, max_landmarks: int, starting_position: np.ndarray, sigma: float):
        step = 6 + 6*max_landmarks
        particles = np.zeros(n_particles * step, dtype=np.float32)

        particles[0::step] = starting_position[0] + np.random.normal(loc=0, scale=sigma, size=n_particles)
        particles[1::step] = starting_position[1] + np.random.normal(loc=0, scale=sigma, size=n_particles)
        particles[2::step] = starting_position[2] + (np.random.normal(loc=0, scale=sigma, size=n_particles) % (2*math.pi))
        particles[3::step] = 1/n_particles
        particles[4::step] = float(max_landmarks)

        return particles