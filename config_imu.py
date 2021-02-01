import numpy as np
from utils import dotify

config = {
    "SEED": 2,
    "N": 8192, # number of particles
    "THREADS": 512, # number threads in a block
    "GPU_HEAP_SIZE_BYTES": 100000 * 1024, # available GPU heap size
    "THRESHOLD": 0.01,
    "sensor": {
        "RANGE": 10,
        "FOV": np.pi,
        "MISS_PROB": 0.05, # probability landmark in range will be missed
        "VARIANCE": [0.2, 0.2],
        "MAX_MEASUREMENTS": 20 # upper bound on the total number of simultaneous measurements
    },
    "ODOMETRY": np.load("fsonline/odom.npy").astype(np.float32)[2000::10],
    "ODOMETRY_VARIANCE": [0.1, 0.1, 0.05],
    "LANDMARKS": np.load("fsonline/track.npy").astype(np.float32), # landmark positions
    "MAX_LANDMARKS": 250 # upper bound on the total number of landmarks in the environment
}

config = dotify(config)

config.sensor.COVARIANCE = \
    np.diag(config.sensor.VARIANCE).astype(np.float32)

config.START_POSITION = config.ODOMETRY[0]

config.PARTICLES_PER_THREAD = config.N // config.THREADS
config.PARTICLE_SIZE = 6 + 7*config.MAX_LANDMARKS
