import math
import time
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
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
from map import compute_map
from vehicle import Vehicle
from stats import Stats
from config_utias_fake_sensor import config


if __name__ == "__main__":
    # visualization
    PLOT = True

    np.random.seed(config.SEED)

    context.set_limit(limit.MALLOC_HEAP_SIZE, config.GPU_HEAP_SIZE_BYTES)

    assert config.THREADS <= 1024 # cannot run more in a single block
    assert config.N >= config.THREADS
    assert config.N % config.THREADS == 0

    particles = FlatParticle.get_initial_particles(config.N, config.MAX_LANDMARKS, config.START_POSITION, sigma=0.2)
    print("Particles memory:", particles.nbytes / 1024, "KB")

    if PLOT:
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        fig.tight_layout()

    sensor = Sensor(
        config.LANDMARKS[:, 1:], [],
        config.sensor.VARIANCE, config.sensor.RANGE,
        config.sensor.FOV, config.sensor.MISS_PROB, 0
    )

    cuda_particles = cuda.mem_alloc(4 * config.N * config.PARTICLE_SIZE)
    scratchpad_block_size = 2 * config.THREADS * config.MAX_LANDMARKS
    cuda_scratchpad = cuda.mem_alloc(4 * scratchpad_block_size)
    cuda_measurements = cuda.mem_alloc(4 * 2 * config.sensor.MAX_MEASUREMENTS)
    cuda_weights = cuda.mem_alloc(8 * config.N)
    cuda_ancestors = cuda.mem_alloc(4 * config.N)
    cuda_ancestors_aux = cuda.mem_alloc(4 * config.N)
    cuda_map = cuda.mem_alloc(4 * 2 * config.MAX_LANDMARKS)
    cuda_map_size = cuda.mem_alloc(4)
    cuda_rescale_sum = cuda.mem_alloc(8)
    cuda_cov = cuda.mem_alloc(4 * 4)
    cuda_mean_position = cuda.mem_alloc(4 * 3)
    cuda_cumsum = cuda.mem_alloc(8 * config.N)
    weights = np.zeros(config.N, dtype=np.float64)
    host_mean_position = np.zeros(3, dtype=np.float32)

    cuda_modules = load_cuda_modules(
        THREADS=config.THREADS,
        PARTICLE_SIZE=config.PARTICLE_SIZE,
        N_PARTICLES=config.N,
        SCRATCHPAD_SIZE=scratchpad_block_size
    )

    cuda.memcpy_htod(cuda_cov, config.sensor.COVARIANCE)
    cuda.memcpy_htod(cuda_particles, particles)

    cuda_modules["predict"].get_function("init_rng")(
        np.int32(config.SEED), block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
    )

    stats = Stats("Loop", "Resample", "Measurement")
    stats.add_pose(config.START_POSITION, config.START_POSITION)
    print("starting..")

    for i, (g, o) in enumerate(zip(config.GROUND_TRUTH, config.CONTROL)):
        stats.start_measuring("Loop")
        print(i)

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

        cuda.memcpy_htod(cuda_measurements, visible_measurements.copy())

        cuda_modules["predict"].get_function("predict_from_model")(
            cuda_particles,
            np.float32(o[2]), np.float32(o[1]),
            np.float32(config.CONTROL_VARIANCE[1]), np.float32(config.CONTROL_VARIANCE[0]),
            np.float32(config.DT),
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )

        cuda_modules["update"].get_function("update")(
            cuda_particles, np.int32(config.N//config.THREADS),
            cuda_scratchpad, np.int32(scratchpad_block_size),
            cuda_measurements,
            np.int32(config.N), np.int32(len(visible_measurements)),
            cuda_cov, np.float32(config.THRESHOLD),
            np.float32(config.sensor.RANGE), np.float32(config.sensor.FOV),
            np.int32(config.MAX_LANDMARKS),
            block=(config.THREADS, 1, 1)
        )

        cuda_modules["rescale"].get_function("sum_weights")(
            cuda_particles, cuda_rescale_sum,
            block=(config.THREADS, 1, 1)
        )

        cuda_modules["rescale"].get_function("divide_weights")(
            cuda_particles, cuda_rescale_sum,
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )

        cuda_modules["weights_and_mean"].get_function("get_mean_position")(
            cuda_particles, cuda_mean_position,
            block=(config.THREADS, 1, 1)
        )

        cuda.memcpy_dtoh(host_mean_position, cuda_mean_position)

        stats.add_pose(g[1:], host_mean_position.copy())


        if PLOT:
            cuda.memcpy_dtoh(particles, cuda_particles)

            ax[0].clear()
            ax[1].clear()

            # ax[0].set_axis_off()
            # ax[1].set_axis_off()
            # ax[0].axis('scaled')
            # ax[1].axis('scaled')

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
            cuda_particles, cuda_weights,
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )
        cuda.memcpy_dtoh(weights, cuda_weights)

        neff = FlatParticle.neff(weights)
        if neff < 0.6*config.N:
            rand = 0.5
            stats.start_measuring("Resample")
            cumsum = np.cumsum(weights)

            cuda.memcpy_htod(cuda_cumsum, cumsum)
            stats.stop_measuring("Resample")

            cuda_modules["resample"].get_function("systematic_resample")(
                cuda_weights, cuda_cumsum, np.float64(rand), cuda_ancestors,
                block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
            )

            cuda_modules["permute"].get_function("reset")(
                cuda_ancestors_aux,
                block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
            )

            cuda_modules["permute"].get_function("compute_positions")(
                cuda_ancestors,
                cuda_ancestors_aux,
                block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
            )

            cuda_modules["permute"].get_function("permute")(
                cuda_ancestors,
                cuda_ancestors_aux,
                np.int32(config.N//config.THREADS), np.int32(config.N),
                block=(config.THREADS, 1, 1)
            )

            cuda_modules["resample"].get_function("copy_inplace")(
                cuda_particles, cuda_ancestors,
                block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
            )

            cuda_modules["resample"].get_function("reset_weights")(
                cuda_particles,
                block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
            )

        stats.stop_measuring("Loop")

    stats.summary()