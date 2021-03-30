import numpy as np
from utils import dotify

START = 1000
T = 1000.0
robot = 1

ground = np.load(f"utias/npy/ground_{robot}_50hz.npy")[START:]
odom = np.load(f"utias/npy/odom_{robot}_50hz.npy")[START:]
measurements = np.load(f"utias/npy/measurements_xy_{robot}_50hz.npy")
landmarks = np.load("utias/landmarks.npy")

ground = ground[ground[:, 0] < T]
odom = odom[odom[:, 0] < T]
measurements = measurements[measurements[:, 0] < T]

config = {
    "SEED": 1,
    "N": 4096,  # number of particles
    "DT": 0.02,
    "THREADS": 512,  # number threads in a block
    "GPU_HEAP_SIZE_BYTES": 100000 * 1024,  # available GPU heap size
    "THRESHOLD": 0.01,
    "sensor": {
        "RANGE": 4,
        "FOV": 0.5*np.pi,
        "MISS_PROB": 0.05,
        "VARIANCE": [0.15, 0.15],
        "MAX_MEASUREMENTS": 20, # upper bound on the total number of simultaneous measurements
        "MEASUREMENTS": measurements.astype(np.float32),
    },
    "CONTROL": odom.astype(np.float32),
    "CONTROL_VARIANCE": [0.015, 0.015],
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
