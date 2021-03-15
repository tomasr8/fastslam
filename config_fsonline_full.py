import json
import numpy as np
from utils import dotify

odometry = np.load("fsonline/full_pipeline/odom.npy").astype(np.float32)

with open("fsonline/full_pipeline/detections_converted.json") as f:
    measurements = json.load(f)

config = {
    "SEED": 2,
    "N": 512, # number of particles
    "THREADS": 512, # number threads in a block
    "GPU_HEAP_SIZE_BYTES": 100000 * 1024, # available GPU heap size
    "THRESHOLD": 0.01,
    "sensor": {
        "RANGE": 30,
        "FOV": np.pi,
        "VARIANCE": [1.0, 1.0],
        "MAX_MEASUREMENTS": 50, # upper bound on the total number of simultaneous measurements
        "MEASUREMENTS": measurements,
    },
    "ODOMETRY": odometry,
    "ODOMETRY_VARIANCE": [0.2, 0.2, 0.05],
    "LANDMARKS": np.load("fsonline/track.npy").astype(np.float32), # landmark positions
    "MAX_LANDMARKS": 250, # upper bound on the total number of landmarks in the environment
    "START_POSITION": odometry[0, :3]
}

config = dotify(config)

config.sensor.COVARIANCE = \
    np.diag(config.sensor.VARIANCE).astype(np.float32)

config.PARTICLES_PER_THREAD = config.N // config.THREADS
config.PARTICLE_SIZE = 6 + 7*config.MAX_LANDMARKS
