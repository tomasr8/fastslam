#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define N_PARTICLES <<N_PARTICLES>>
#define PARTICLE_SIZE <<PARTICLE_SIZE>>
#define THREADS <<THREADS>>

__device__ float* get_particle(float *particles, int i) {
    return (particles + PARTICLE_SIZE*i);
}

// /*
//  * Extracts the particle weights and computes the estimated position.
//  * This code is not parallelized since it is reasonobly fast.
//  */
// __global__ void get_weights_and_mean_position(float *particles, int n_particles, float *weights, float *mean) {
//     float x = 0;
//     float y = 0;
//     float theta = 0;

//     for(int i = 0; i < n_particles; i++) {
//         float *particle = get_particle(particles, i);
//         weights[i] = particle[3];
//         x += particle[3] * particle[0];
//         y += particle[3] * particle[1];
//         theta += particle[3] * particle[2];
//     }

//     mean[0] = x;
//     mean[1] = y;
//     mean[2] = theta;
// }


/*
 * Extracts weights from particles.
 */
 __global__ void get_weights(float *particles, double *weights) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    float *particle = get_particle(particles, i);
    weights[i] = (double)particle[3];
}


/*
 * Calculates the mean position of all particles.
 * Needs to run in a single block.
 */
 __global__ void get_mean_position(float *particles, float *mean) {
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    int idx = block_id * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    float x = 0;
    float y = 0;
    float theta = 0;

    for (int i = idx; i < N_PARTICLES; i += THREADS) {
        float *particle = get_particle(particles, i);
        x += particle[3] * particle[0];
        y += particle[3] * particle[1];
        theta += particle[3] * particle[2];
    }

    __shared__ float r_x[THREADS];
    __shared__ float r_y[THREADS];
    __shared__ float r_theta[THREADS];

    r_x[idx] = x;
    r_y[idx] = y;
    r_theta[idx] = theta;

    __syncthreads();

    for (int size = THREADS/2; size>0; size/=2) {
        if (idx<size) {
            r_x[idx] += r_x[idx+size];
            r_y[idx] += r_y[idx+size];
            r_theta[idx] += r_theta[idx+size];
        }
        __syncthreads();
    }

    if (idx == 0) {
        mean[0] = r_x[0];
        mean[1] = r_y[0];
        mean[2] = r_theta[0];
    }
}