import numpy as np
from utils import dotify
from cuda.update_jacobian_dist import load_cuda_modules

config = {
    "SEED": 1,
    "N": 4096,  # number of particles
    "DT": 0.5,
    "THREADS": 512,  # number threads in a block
    "GPU_HEAP_SIZE_BYTES": 100000 * 1024,  # available GPU heap size
    "THRESHOLD": 1.2,
    "sensor": {
        "RANGE": 5,
        "FOV": np.pi,
        "MISS_PROB": 0,  # probability landmark in range will be missed
        "VARIANCE": [0.05, 0.003],
        "MAX_MEASUREMENTS": 50  # upper bound on the total number of simultaneous measurements
    },
    # "gps": {
    #     "VARIANCE": [0.1, 0.1, 0.01],
    #     "RATE": 50
    # },
    "CONTROL": np.vstack((
        np.tile([0.0, 0.0], (15, 1)),
        np.tile([0.0, 0.5], (70, 1)),
        np.tile([0.21, 0.0], (14, 1)),
        np.tile([0.0, 0.5], (70, 1)),
        np.tile([0.21, 0.0], (14, 1)),
        np.tile([0.0, 0.5], (70, 1)),
        np.tile([0.21, 0.0], (14, 1)),
        np.tile([0.0, 0.5], (70, 1))
    )),
    "CONTROL_VARIANCE": [0.02, 0.1],
    "LANDMARKS": np.loadtxt("landmarks_square.txt").astype(np.float32),  # landmark positions
    "MAX_LANDMARKS": 250,  # upper bound on the total number of landmarks in the environment
    "START_POSITION": np.array([1, 1, 0], dtype=np.float32)
}

config = dotify(config)

config.sensor.COVARIANCE = \
    np.diag(config.sensor.VARIANCE).astype(np.float32)

config.PARTICLES_PER_THREAD = config.N // config.THREADS
config.PARTICLE_SIZE = 6 + 7*config.MAX_LANDMARKS

config.modules = load_cuda_modules(
    THREADS=config.THREADS,
    PARTICLE_SIZE=config.PARTICLE_SIZE,
    N_PARTICLES=config.N
)