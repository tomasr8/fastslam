import numpy as np
from utils import dotify

START = 2000
ground = np.load("utias/ground_10hz.npy")[START:]
odom = np.load("utias/odom_10hz.npy")[START:]
landmarks = np.load("utias/landmarks.npy")

config = {
    "SEED": 2,
    "N": 512,  # number of particles
    "DT": 0.1,
    "THREADS": 512,  # number threads in a block
    "GPU_HEAP_SIZE_BYTES": 100000 * 1024,  # available GPU heap size
    "THRESHOLD": 0.01,
    "sensor": {
        "RANGE": 2.5,
        "FOV": 0.7*np.pi,
        "MISS_PROB": 0.3,
        "VARIANCE": [0.05, 0.05],
        "MAX_MEASUREMENTS": 20 # upper bound on the total number of simultaneous measurements
    },
    "CONTROL": odom.astype(np.float32),
    "CONTROL_VARIANCE": [0.01, 0.01],
    "GROUND_TRUTH": ground.astype(np.float32),
    "LANDMARKS": landmarks.astype(np.float32),  # landmark positions
    "MAX_LANDMARKS": 30,  # upper bound on the total number of landmarks in the environment
    "START_POSITION": ground[0, 1:].astype(np.float32),
    
}

config = dotify(config)

config.sensor.COVARIANCE = \
    np.diag(config.sensor.VARIANCE).astype(np.float32)

config.PARTICLES_PER_THREAD = config.N // config.THREADS
config.PARTICLE_SIZE = 6 + 7*config.MAX_LANDMARKS
