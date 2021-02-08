import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_confidence_ellipse, plot_connections, plot_history, plot_landmarks, plot_sensor_fov
from particle3 import FlatParticle

FLOAT = 4
DOUBLE = 8


class CUDAMemory:
    def __init__(self, config):
        self.particles = cuda.mem_alloc(FLOAT * config.N * config.PARTICLE_SIZE)

        self.scratchpad_block_size = 2 * config.THREADS * config.MAX_LANDMARKS
        self.scratchpad = cuda.mem_alloc(FLOAT * self.scratchpad_block_size)

        self.measurements = cuda.mem_alloc(FLOAT * 2 * config.sensor.MAX_MEASUREMENTS)
        self.weights = cuda.mem_alloc(DOUBLE * config.N)
        self.ancestors = cuda.mem_alloc(FLOAT * config.N)
        self.ancestors_aux = cuda.mem_alloc(FLOAT * config.N)
        self.rescale_sum = cuda.mem_alloc(DOUBLE)
        self.cov = cuda.mem_alloc(FLOAT * 4)
        self.mean_position = cuda.mem_alloc(FLOAT * 3)
        self.cumsum = cuda.mem_alloc(8 * config.N)
        self.c = cuda.mem_alloc(FLOAT * config.N)
        self.d = cuda.mem_alloc(FLOAT * config.N)


def get_pose_estimate(modules, config, memory: CUDAMemory):
    estimate = np.zeros(3, dtype=np.float32)

    modules["weights_and_mean"].get_function("get_mean_position")(
        memory.particles, memory.mean_position,
        block=(config.THREADS, 1, 1)
    )

    cuda.memcpy_dtoh(estimate, memory.mean_position)
    return estimate


def rescale(modules, config, memory: CUDAMemory):
    modules["rescale"].get_function("sum_weights")(
        memory.particles, memory.rescale_sum,
        block=(config.THREADS, 1, 1)
    )

    modules["rescale"].get_function("divide_weights")(
        memory.particles, memory.rescale_sum,
        block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
    )


def resample(modules, config, weights, memory: CUDAMemory, rand):
    cumsum = np.cumsum(weights)

    cuda.memcpy_htod(memory.cumsum, cumsum)

    modules["resample"].get_function("systematic_resample")(
        memory.weights, memory.cumsum, np.float64(rand), memory.ancestors,
        block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
    )

    modules["permute_ref"].get_function("reset")(memory.d, np.int32(config.N), block=(
        config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1))
    modules["permute_ref"].get_function("prepermute")(memory.ancestors, memory.d, block=(
        config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1))
    modules["permute_ref"].get_function("permute")(memory.ancestors, memory.c, memory.d, np.int32(
        config.N), block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1))
    modules["permute_ref"].get_function("write_to_c")(memory.ancestors, memory.c, memory.d, block=(
        config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1))

    modules["resample"].get_function("copy_inplace")(
        memory.particles, memory.c,
        block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
    )

    modules["resample"].get_function("reset_weights")(
        memory.particles,
        block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
    )


# def resample(modules, config, weights, cuda_particles, cuda_weights, cuda_cumsum, cuda_ancestors, cuda_ancestors_aux, rand):
#     cumsum = np.cumsum(weights)

#     cuda.memcpy_htod(cuda_cumsum, cumsum)

#     modules["resample"].get_function("systematic_resample")(
#         cuda_weights, cuda_cumsum, np.float64(rand), cuda_ancestors,
#         block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
#     )


#     modules.get_function("reset")(cuda_d, np.int32(N), block=(MAX_BLOCK_SIZE, 1, 1), grid=(N//MAX_BLOCK_SIZE, 1, 1))
#     modules.get_function("prepermute")(cuda_ancestors, cuda_d, block=(MAX_BLOCK_SIZE, 1, 1), grid=(N//MAX_BLOCK_SIZE, 1, 1))
#     modules.get_function("permute_reference")(cuda_ancestors, cuda_c, cuda_d, np.int32(N), block=(MAX_BLOCK_SIZE, 1, 1), grid=(N//MAX_BLOCK_SIZE, 1, 1))
#     modules.get_function("write_to_c")(cuda_ancestors, cuda_c, cuda_d, block=(MAX_BLOCK_SIZE, 1, 1), grid=(N//MAX_BLOCK_SIZE, 1, 1))


#     modules["permute"].get_function("reset")(
#         cuda_ancestors_aux,
#         block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
#     )

#     modules["permute"].get_function("compute_positions")(
#         cuda_ancestors,
#         cuda_ancestors_aux,
#         block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
#     )

#     modules["permute"].get_function("permute")(
#         cuda_ancestors,
#         cuda_ancestors_aux,
#         np.int32(config.N//config.THREADS), np.int32(config.N),
#         block=(config.THREADS, 1, 1)
#     )

#     modules["resample"].get_function("copy_inplace")(
#         cuda_particles, cuda_ancestors,
#         block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
#     )

#     modules["resample"].get_function("reset_weights")(
#         cuda_particles,
#         block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
#     )


# def update_plot(ax, modules, config, particles, ):
#     ax[0].clear()
#     ax[1].clear()
#     ax[0].set_axis_off()
#     ax[1].set_axis_off()
#     ax[0].axis('scaled')
#     ax[1].axis('scaled')

#     plot_sensor_fov(ax[0], vehicle.position, config.sensor.RANGE, config.sensor.FOV)
#     plot_sensor_fov(ax[1], vehicle.position, config.sensor.RANGE, config.sensor.FOV)

#     if(visible_measurements.size != 0):
#         plot_connections(ax[0], vehicle.position, visible_measurements + vehicle.position[:2])

#     plot_landmarks(ax[0], config.LANDMARKS, color="blue", zorder=100)
#     plot_landmarks(ax[0], out_of_range_landmarks, color="black", zorder=101)
#     plot_history(ax[0], stats.ground_truth_path, color='green')
#     plot_history(ax[0], stats.predicted_path, color='orange')
#     plot_particles_weight(ax[0], particles)
#     if(visible_measurements.size != 0):
#         plot_measurement(ax[0], vehicle.position[:2], visible_measurements, color="orange", zorder=103)

#     plot_landmarks(ax[0], missed_landmarks, color="red", zorder=102)

#     best = np.argmax(FlatParticle.w(particles))
#     plot_landmarks(ax[1], config.LANDMARKS, color="black")
#     covariances = FlatParticle.get_covariances(particles, best)

#     plot_map(ax[1], FlatParticle.get_landmarks(particles, best), color="orange", marker="o")

#     for i, landmark in enumerate(FlatParticle.get_landmarks(particles, best)):
#         plot_confidence_ellipse(ax[1], landmark, covariances[i], n_std=3)

#     plt.pause(0.01)
