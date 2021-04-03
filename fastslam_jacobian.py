import math
import time
import json
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

from cuda.update_jacobian_dist import load_cuda_modules
from sensor import Sensor
from vehicle import Vehicle
from stats import Stats
from common import CUDAMemory, resample, rescale, get_pose_estimate
from utils import repeat


def run_SLAM(config, plot=False, seed=None):
    if seed is None:
        np.random.seed(config.SEED)
    else:
        np.random.seed(seed)

    assert config.THREADS <= 1024  # cannot run more in a single block
    assert config.N >= config.THREADS
    assert config.N % config.THREADS == 0

    particles = FlatParticle.get_initial_particles(config.N, config.MAX_LANDMARKS, config.START_POSITION, sigma=0.2)
    print("Particles memory:", particles.nbytes / 1024, "KB")

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        ax[0].axis('scaled')
        ax[1].axis('scaled')

    sensor = Sensor(
        config.LANDMARKS, [],
        config.sensor.VARIANCE, config.sensor.RANGE,
        config.sensor.FOV, config.sensor.MISS_PROB, 0, rb=True
    )

    vehicle = Vehicle(config.START_POSITION, config.CONTROL_VARIANCE, dt=config.DT)
    cuda_modules = config.modules

    memory = CUDAMemory(config)
    weights = np.zeros(config.N, dtype=np.float64)

    cuda.memcpy_htod(memory.cov, config.sensor.COVARIANCE)
    cuda.memcpy_htod(memory.particles, particles)

    cuda_modules["predict"].get_function("init_rng")(
        np.int32(config.SEED), block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
    )


    stats = Stats("Loop", "Measurement")
    stats.add_pose(config.START_POSITION.tolist(), config.START_POSITION.tolist())
    print("starting..")


    for i in range(config.CONTROL.shape[0]):
        stats.start_measuring("Loop")
        print(i)

        stats.start_measuring("Measurement")
        vehicle.move_noisy(config.CONTROL[i])

        measurements = sensor.get_noisy_measurements(vehicle.position)
        visible_measurements = measurements["observed"]
        missed_landmarks = measurements["missed"]
        out_of_range_landmarks = measurements["outOfRange"]

        stats.stop_measuring("Measurement")

        cuda.memcpy_htod(memory.measurements, visible_measurements)

        if "gps" in config and i % config.gps.RATE == 0:
            cuda_modules["predict"].get_function("predict_from_imu")(
                memory.particles,
                np.float32(vehicle.position[0]), np.float32(vehicle.position[1]), np.float32(vehicle.position[2]),
                np.float32(config.gps.VARIANCE[0]), np.float32(config.gps.VARIANCE[1]), np.float32(config.gps.VARIANCE[2]),
                block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
            )
        else:
            cuda_modules["predict"].get_function("predict_from_model")(
                memory.particles,
                np.float32(config.CONTROL[i, 0]), np.float32(config.CONTROL[i, 1]),
                np.float32(config.CONTROL_VARIANCE[0]), np.float32(config.CONTROL_VARIANCE[1]),
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

        stats.add_pose(vehicle.position.tolist(), estimate.tolist())

        if plot:
            cuda.memcpy_dtoh(particles, memory.particles)

            visible_measurements = [[r*np.cos(b + vehicle.position[2]), r*np.sin(b + vehicle.position[2])]
                                    for (r, b) in visible_measurements]

            visible_measurements = np.array(visible_measurements)

            ax[0].clear()
            ax[1].clear()
            # ax[0].set_axis_off()
            # ax[1].set_axis_off()
            ax[0].axis('scaled')
            ax[1].axis('scaled')

            plot_sensor_fov(ax[0], vehicle.position, config.sensor.RANGE, config.sensor.FOV)
            plot_sensor_fov(ax[1], vehicle.position, config.sensor.RANGE, config.sensor.FOV)

            if(visible_measurements.size != 0):
                plot_connections(ax[0], vehicle.position, visible_measurements + vehicle.position[:2])

            plot_landmarks(ax[0], config.LANDMARKS, color="blue", zorder=100)
            plot_landmarks(ax[0], out_of_range_landmarks, color="black", zorder=101)
            plot_history(ax[0], stats.ground_truth_path, color='green')
            plot_history(ax[0], stats.predicted_path, color='orange')
            plot_particles_weight(ax[0], particles)
            if(visible_measurements.size != 0):
                plot_measurement(ax[0], vehicle.position[:2], visible_measurements, color="orange", zorder=103)

            plot_landmarks(ax[0], missed_landmarks, color="red", zorder=102)

            best = np.argmax(FlatParticle.w(particles))
            plot_landmarks(ax[1], config.LANDMARKS, color="black")
            covariances = FlatParticle.get_covariances(particles, best)

            plot_map(ax[1], FlatParticle.get_landmarks(particles, best), color="orange", marker="o")

            for i, landmark in enumerate(FlatParticle.get_landmarks(particles, best)):
                plot_confidence_ellipse(ax[1], landmark, covariances[i], n_std=3)

            plt.pause(0.001)


        if i == config.CONTROL.shape[0]-1:
            cuda.memcpy_dtoh(particles, memory.particles)
            best = np.argmax(FlatParticle.w(particles))
            best_covariances = FlatParticle.get_covariances(particles, best)
            best_landmarks = FlatParticle.get_landmarks(particles, best)


        cuda_modules["weights_and_mean"].get_function("get_weights")(
            memory.particles, memory.weights,
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )
        cuda.memcpy_dtoh(weights, memory.weights)

        # if i == config.CONTROL.shape[0]-1:
        #     cuda.memcpy_dtoh(particles, memory.particles)
        #     np.save("particles.npy", particles)

        neff = FlatParticle.neff(weights)
        if neff < 0.6*config.N:
            resample(cuda_modules, config, weights, memory, 0.5)

        stats.stop_measuring("Loop")


    if not plot:
        fig, ax = plt.subplots()
        plot_history(ax, stats.ground_truth_path, color='green')
        plot_history(ax, stats.predicted_path, color='orange')
        plot_landmarks(ax, config.LANDMARKS, color="blue")
        plot_map(ax, best_landmarks, color="orange", marker="o")
        for i, landmark in enumerate(best_landmarks):
            plot_confidence_ellipse(ax, landmark, best_covariances[i], n_std=3)

        plt.savefig(f"figs_jacobi_dist/{seed}.png")

    # output = {
    #     "ground": stats.ground_truth_path,
    #     "predicted": stats.predicted_path,
    #     "landmarks": config.LANDMARKS.tolist()
    # }

    # with open("out_jacobian.json", "w") as f:
    #     json.dump(output, f)

    stats.summary()
    return stats.mean_path_deviation()


if __name__ == "__main__":
    # from config_jacobian import config
    from config_jacobian_square import config
    context.set_limit(limit.MALLOC_HEAP_SIZE, config.GPU_HEAP_SIZE_BYTES)

    run_SLAM(config, plot=True)

    # context.set_limit(limit.MALLOC_HEAP_SIZE, config.GPU_HEAP_SIZE_BYTES)
    # print(repeat(run_SLAM, seeds=np.arange(100)))