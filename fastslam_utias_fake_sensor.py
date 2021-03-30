import math
import time
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import json
from plotting import (
    plot_connections, plot_history, plot_landmarks, plot_measurement,
    plot_particles_weight, plot_particles_grey, plot_confidence_ellipse,
    plot_sensor_fov, plot_map
)
from particle3 import FlatParticle

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.driver import limit

from cuda.update3 import load_cuda_modules
from sensor import Sensor
from vehicle import Vehicle
from stats import Stats
from common import CUDAMemory, resample, rescale, get_pose_estimate
from config_utias_fake_sensor import config

def run_SLAM(plot=False):
    np.random.seed(config.SEED)

    context.set_limit(limit.MALLOC_HEAP_SIZE, config.GPU_HEAP_SIZE_BYTES)

    assert config.THREADS <= 1024 # cannot run more in a single block
    assert config.N >= config.THREADS
    assert config.N % config.THREADS == 0

    particles = FlatParticle.get_initial_particles(config.N, config.MAX_LANDMARKS, config.START_POSITION, sigma=0.2)
    print("Particles memory:", particles.nbytes / 1024, "KB")

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(5, 10))
        fig.tight_layout()

    memory = CUDAMemory(config)
    weights = np.zeros(config.N, dtype=np.float64)

    cuda_modules = load_cuda_modules(
        THREADS=config.THREADS,
        PARTICLE_SIZE=config.PARTICLE_SIZE,
        N_PARTICLES=config.N
    )

    sensor = Sensor(
        config.LANDMARKS[:, 1:], [],
        config.sensor.VARIANCE, config.sensor.RANGE,
        config.sensor.FOV, config.sensor.MISS_PROB, 0
    )

    cuda.memcpy_htod(memory.cov, config.sensor.COVARIANCE)
    cuda.memcpy_htod(memory.particles, particles)

    cuda_modules["predict"].get_function("init_rng")(
        np.int32(config.SEED), block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
    )


    stats = Stats("Loop", "Measurement")
    stats.add_pose(config.START_POSITION.tolist(), config.START_POSITION.tolist())
    print("starting..")

    for i, (g, o) in enumerate(zip(config.GROUND_TRUTH, config.CONTROL)):
        stats.start_measuring("Loop")

        stats.start_measuring("Measurement")

        # print(g)
        pose = g[1:]

        if i % 10 == 0:
            measurements = sensor.get_noisy_measurements(pose)
            visible_measurements = measurements["observed"]
            missed_landmarks = measurements["missed"]
            out_of_range_landmarks = measurements["outOfRange"]
        else:
            visible_measurements = np.zeros(shape=(0, 2), dtype=np.float32)
            missed_landmarks = config.LANDMARKS[:, 1:]
            out_of_range_landmarks = np.zeros(shape=(0, 2), dtype=np.float32)


        stats.stop_measuring("Measurement")

        cuda.memcpy_htod(memory.measurements, visible_measurements.copy())

        cuda_modules["predict"].get_function("predict_from_model")(
            memory.particles,
            np.float32(o[2]), np.float32(o[1]),
            np.float32(config.CONTROL_VARIANCE[1]), np.float32(config.CONTROL_VARIANCE[0]),
            np.float32(config.DT),
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )

        cuda_modules["update"].get_function("update")(
            memory.particles, np.int32(config.N//config.THREADS),
            memory.scratchpad, np.int32(memory.scratchpad_block_size),
            memory.measurements,
            np.int32(config.N), np.int32(len(visible_measurements)),
            memory.cov, np.float32(config.THRESHOLD),
            np.float32(config.sensor.RANGE), np.float32(config.sensor.FOV),
            np.int32(config.MAX_LANDMARKS),
            block=(config.THREADS, 1, 1)
        )

        rescale(cuda_modules, config, memory)
        estimate =  get_pose_estimate(cuda_modules, config, memory)

        stats.add_pose(g[1:].tolist(), estimate.tolist())

        if plot and i % 100 == 0:
            cuda.memcpy_dtoh(particles, memory.particles)

            ax[0].clear()
            ax[1].clear()

            plot_sensor_fov(ax[0], g[1:], config.sensor.RANGE, config.sensor.FOV)
            plot_sensor_fov(ax[1], g[1:], config.sensor.RANGE, config.sensor.FOV)

            if(visible_measurements.size != 0):
                plot_connections(ax[0], g[1:], visible_measurements + g[1:3])

            plot_landmarks(ax[0], config.LANDMARKS[:, 1:], color="blue", zorder=100)
            plot_landmarks(ax[0], out_of_range_landmarks, color="black", zorder=101)
            plot_history(ax[0], stats.ground_truth_path[::20], color='green')
            plot_history(ax[0], stats.predicted_path[::20], color='orange')
            plot_particles_weight(ax[0], particles)

            if(visible_measurements.size != 0):
                plot_measurement(ax[0], g[1:3], visible_measurements, color="orange", zorder=103)

            plot_landmarks(ax[0], missed_landmarks, color="red", zorder=102)

            best = np.argmax(FlatParticle.w(particles))
            plot_landmarks(ax[1], config.LANDMARKS[:, 1:], color="black")
            covariances = FlatParticle.get_covariances(particles, best)

            plot_map(ax[1], FlatParticle.get_landmarks(particles, best), color="orange", marker="o")

            for i, landmark in enumerate(FlatParticle.get_landmarks(particles, best)):
                plot_confidence_ellipse(ax[1], landmark, covariances[i], n_std=3)

            plt.pause(0.001)


        cuda_modules["weights_and_mean"].get_function("get_weights")(
            memory.particles, memory.weights,
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )
        cuda.memcpy_dtoh(weights, memory.weights)

        neff = FlatParticle.neff(weights)
        if neff < 0.6*config.N:
            resample(cuda_modules, config, weights, memory, 0.5)

        stats.stop_measuring("Loop")

    stats.summary()

    output = {
        "ground": stats.ground_truth_path,
        "predicted": stats.predicted_path,
        "landmarks": config.LANDMARKS[:, 1:].tolist()
    }

    with open("out.json", "w") as f:
        json.dump(output, f)

    return stats.mean_path_deviation()


if __name__ == "__main__":
    run_SLAM(plot=False)