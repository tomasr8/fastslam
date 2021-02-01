import math
import time
from typing import List
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

from plotting import (
    plot_connections, plot_history, plot_landmarks, plot_measurement,
    plot_particles_weight, plot_particles_grey, plot_confidence_ellipse,
    plot_sensor_fov, plot_map
)
from particle3 import FlatParticle, systematic_resample

from multiprocessing import Process

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.driver import limit

from cuda.update3 import load_cuda_modules
from sensor import Sensor
from map import compute_map
from stats import Stats, mean_path_deviation


if __name__ == "__main__":
    # randomness
    np.random.seed(2)

    # visualization
    PLOT = False

    # simulation
    N = 8192  # number of particles
    SIM_LENGTH = 200  # number of simulation steps
    MAX_RANGE = 10  # max range of sensor
    MAX_FOV = (1)*np.pi
    MISS_PROB = 0.05  # probability landmark in range will be missed
    MAX_LANDMARKS = 250  # upper bound on the total number of landmarks in the environment
    MAX_MEASUREMENTS = 20  # upper bound on the total number of simultaneous measurements
    odom_variance = [0.1, 0.1, 0.05]
    measurement_variance = [0.2, 0.2]
    measurement_covariance = np.float32([
        measurement_variance[0], 0,
        0, measurement_variance[1]
    ])
    THRESHOLD = 0.01 

    landmarks = np.load("fsonline/track.npy").astype(np.float32)  # landmark positions
    landmarks = landmarks[:, [0, 1]]
    odom = np.load("fsonline/odom.npy")

    odom = odom[2000:]
    odom = odom[::10]
    odom[:, 2] += np.pi/2

    start_position = np.array(odom[0], dtype=np.float32)  # starting position of the car

    # GPU
    context.set_limit(limit.MALLOC_HEAP_SIZE, 100000 * 1024)  # heap size available to the GPU threads
    THREADS = 512  # number of GPU threads
    assert THREADS <= 1024 # cannot run more in a single block
    assert N >= THREADS
    assert N % THREADS == 0
    BLOCK_SIZE = N//THREADS  # number of particles per thread
    PARTICLE_SIZE = 6 + 7*MAX_LANDMARKS

    particles = FlatParticle.get_initial_particles(N, MAX_LANDMARKS, start_position, sigma=0.2)
    print("Particles memory:", particles.nbytes / 1024, "KB")

    if PLOT:
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        ax[0].axis('scaled')
        ax[1].axis('scaled')


    real_position_history = [start_position]
    predicted_position_history = [start_position]
    weights_history = [N]

    sensor = Sensor(landmarks, [], measurement_variance, MAX_RANGE, MAX_FOV, MISS_PROB, 0)

    cuda_particles = cuda.mem_alloc(4 * N * (6 + 7*MAX_LANDMARKS))
    scratchpad_block_size = 2 * THREADS * MAX_LANDMARKS
    cuda_scratchpad = cuda.mem_alloc(4 * scratchpad_block_size)
    cuda_measurements = cuda.mem_alloc(4 * 2 * MAX_MEASUREMENTS)
    cuda_weights = cuda.mem_alloc(8 * N)
    cuda_ancestors = cuda.mem_alloc(4 * N)
    cuda_ancestors_aux = cuda.mem_alloc(4 * N)
    cuda_map = cuda.mem_alloc(4 * 2 * MAX_LANDMARKS)
    cuda_map_size = cuda.mem_alloc(4)
    cuda_rescale_sum = cuda.mem_alloc(4)
    cuda_cov = cuda.mem_alloc(4 * 4)
    cuda_mean_position = cuda.mem_alloc(4 * 3)
    host_weights = np.zeros(N, dtype=np.float64)
    host_mean_position = np.zeros(3, dtype=np.float32)

    cuda_modules = load_cuda_modules(
        THREADS=THREADS,
        PARTICLE_SIZE=PARTICLE_SIZE,
        N_PARTICLES=N,
        BLOCK_SIZE=BLOCK_SIZE,
        SCRATCHPAD_SIZE=scratchpad_block_size
    )

    cuda.memcpy_htod(cuda_cov, measurement_covariance)
    cuda.memcpy_htod(cuda_particles, particles)

    movement_seed = 2
    cuda_modules["predict"].get_function("init_rng")(
        np.int32(movement_seed), block=(THREADS, 1, 1), grid=(N//THREADS, 1, 1)
    )


    stats = Stats("Loop", "Resample", "Measurement", "1st", "2nd")
    print("starting..")

    for i in range(odom.shape[0]):
        stats.start_measuring("Loop")
        stats.start_measuring("1st")
        print(i)

        stats.start_measuring("Measurement")
        pose = odom[i]
        real_position_history.append(pose)

        measurements = sensor.get_noisy_measurements(pose)
        visible_measurements = measurements["observed"]
        missed_landmarks = measurements["missed"]
        out_of_range_landmarks = measurements["outOfRange"]

        measured_pose = [
            pose[0] + np.random.normal(0, odom_variance[0]),
            pose[1] + np.random.normal(0, odom_variance[1]),
            pose[2] + np.random.normal(0, odom_variance[2])
        ]

        stats.stop_measuring("Measurement")

        cuda.memcpy_htod(cuda_measurements, visible_measurements)

        cuda_modules["predict"].get_function("predict_from_imu")(
            cuda_particles, np.int32(BLOCK_SIZE), np.int32(N),
            np.float32(measured_pose[0]), np.float32(measured_pose[1]), np.float32(measured_pose[2]),
            np.float32(odom_variance[0]), np.float32(odom_variance[1]), np.float32(odom_variance[2]),
            block=(THREADS, 1, 1), grid=(N//THREADS, 1, 1)
        )

        cuda_modules["update"].get_function("update")(
            cuda_particles, np.int32(BLOCK_SIZE),
            cuda_scratchpad, np.int32(scratchpad_block_size),
            cuda_measurements,
            np.int32(N), np.int32(len(visible_measurements)),
            cuda_cov, np.float32(THRESHOLD), np.float32(MAX_RANGE), np.float32(MAX_FOV), np.int32(MAX_LANDMARKS),
            block=(THREADS, 1, 1)
        )

        cuda_modules["rescale"].get_function("sum_weights")(
            cuda_particles, cuda_rescale_sum,
            block=(THREADS, 1, 1)
        )

        cuda_modules["rescale"].get_function("divide_weights")(
            cuda_particles, cuda_rescale_sum,
            block=(THREADS, 1, 1), grid=(N//THREADS, 1, 1)
        )

        cuda_modules["weights_and_mean"].get_function("get_weights")(
            cuda_particles, cuda_weights,
            block=(THREADS, 1, 1), grid=(N//THREADS, 1, 1)
        )

        cuda_modules["weights_and_mean"].get_function("get_mean_position")(
            cuda_particles, cuda_mean_position,
            block=(THREADS, 1, 1)
        )

        cuda.memcpy_dtoh(host_weights, cuda_weights)
        cuda.memcpy_dtoh(host_mean_position, cuda_mean_position)

        predicted_position_history.append(host_mean_position.copy())


        # cuda_modules["kmeans"].get_function("initialize_centroids")(
        #     cuda_old_particles, cuda_weights, np.int32(N), cuda_map, cuda_map_size,
        #     block=(1, 1, 1)
        # )

        # n_centroids = np.zeros(1, dtype=np.int32)
        # cuda.memcpy_dtoh(n_centroids, cuda_map_size)
        # n_centroids = int(n_centroids[0])

        # cuda_modules["kmeans"].get_function("relabel")(
        #     cuda_old_particles, cuda_new_particles, cuda_map,
        #     np.int32(BLOCK_SIZE), np.int32(N), np.int32(n_centroids),
        #     block=(THREADS, 1, 1), grid=(N//THREADS, 1, 1)
        # )

        # cuda_modules["kmeans"].get_function("compute_centroids")(
        #     cuda_old_particles, cuda_new_particles, cuda_map,
        #     np.int32(N), np.int32(n_centroids),
        #     block=(n_centroids, 1, 1)
        # )

        # centroids = np.zeros(MAX_LANDMARKS*2, dtype=np.float32)
        # cuda.memcpy_dtoh(centroids, cuda_map)
        # centroids = centroids.reshape((MAX_LANDMARKS, 2))[:n_centroids]

        if PLOT:
            cuda.memcpy_dtoh(particles, cuda_particles)

            ax[0].clear()
            ax[1].clear()
            # ax[0].set_xlim([pose[0]-10, pose[0]+10])
            # ax[0].set_ylim([pose[1]-10, pose[1]+10])
            # ax[1].set_xlim([pose[0]-10, pose[0]+10])
            # ax[1].set_ylim([pose[1]-10, pose[1]+10])

            ax[0].set_xlim([-160, 10])
            ax[0].set_ylim([-30, 50])
            ax[1].set_xlim([-160, 10])
            ax[1].set_ylim([-30, 50])
            ax[0].set_axis_off()
            ax[1].set_axis_off()

            plot_sensor_fov(ax[0], pose, MAX_RANGE, MAX_FOV)
            plot_sensor_fov(ax[1], pose, MAX_RANGE, MAX_FOV)

            if(visible_measurements.size != 0):
                plot_connections(ax[0], pose, visible_measurements + pose[:2])

            plot_landmarks(ax[0], landmarks, color="blue", zorder=100)
            plot_landmarks(ax[0], out_of_range_landmarks, color="black", zorder=101)
            plot_history(ax[0], real_position_history, color='green')
            plot_history(ax[0], predicted_position_history, color='orange')
            plot_particles_weight(ax[0], particles)
            if(visible_measurements.size != 0):
                plot_measurement(ax[0], pose[:2], visible_measurements, color="orange", zorder=103)

            plot_landmarks(ax[0], missed_landmarks, color="red", zorder=102)

            best = np.argmax(FlatParticle.w(particles))
            plot_landmarks(ax[1], landmarks, color="black")
            # plot_landmarks(ax[1], FlatParticle.get_landmarks(particles, best), color="orange")
            covariances = FlatParticle.get_covariances(particles, best)

            # n = 200
            # cmap = plt.cm.get_cmap("hsv", n)
            # print("n_landmarks:", [len(FlatParticle.get_landmarks(particles, i)) for i in np.argsort(-FlatParticle.w(particles))[:n]])
            # for n, i in enumerate(np.argsort(-FlatParticle.w(particles))[:n]):
            #     plot_map(ax[1], FlatParticle.get_landmarks(particles, i), color=cmap(n), marker=".")

            # plt.pause(5)

            # centroids = compute_map(particles)
            # plot_map(ax[1], centroids, color="orange", marker="o")

            # for i, landmark in enumerate(centroids):
            #     plot_confidence_ellipse(ax[1], landmark, covariances[i], n_std=3)

            plot_map(ax[1], FlatParticle.get_landmarks(particles, best), color="orange", marker="o")

            for i, landmark in enumerate(FlatParticle.get_landmarks(particles, best)):
                plot_confidence_ellipse(ax[1], landmark, covariances[i], n_std=3)

            plt.pause(0.01)



        stats.stop_measuring("1st")
        stats.start_measuring("2nd")


        neff = FlatParticle.neff(host_weights)
        if neff < 0.6*N:
            stats.start_measuring("Resample")
            ancestors = systematic_resample(host_weights)

            cuda.memcpy_htod(cuda_ancestors, ancestors)
            stats.stop_measuring("Resample")

            cuda_modules["permute"].get_function("reset")(
                cuda_ancestors_aux,
                block=(THREADS, 1, 1), grid=(N//THREADS, 1, 1)
            )

            cuda_modules["permute"].get_function("compute_positions")(
                cuda_ancestors,
                cuda_ancestors_aux,
                block=(THREADS, 1, 1), grid=(N//THREADS, 1, 1)
            )

            cuda_modules["permute"].get_function("permute")(
                cuda_ancestors,
                cuda_ancestors_aux,
                np.int32(N//THREADS), np.int32(N),
                block=(THREADS, 1, 1)
            )

            cuda_modules["resample"].get_function("resample_inplace")(
                cuda_particles, cuda_ancestors,
                block=(THREADS, 1, 1), grid=(N//THREADS, 1, 1)
            )

            cuda_modules["resample"].get_function("reset_weights")(
                cuda_particles,
                block=(THREADS, 1, 1), grid=(N//THREADS, 1, 1)
            )

        weights_history.append(neff)
        stats.stop_measuring("2nd")
        stats.stop_measuring("Loop")


    stats.summary()

    deviation = mean_path_deviation(real_position_history, predicted_position_history)
    print(f"Mean path deviation: {deviation}")