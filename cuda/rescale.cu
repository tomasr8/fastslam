#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define PARTICLE_SIZE <<PARTICLE_SIZE>>
#define N_PARTICLES <<N_PARTICLES>>
#define THREADS <<THREADS>>

__device__ float* get_particle(float *particles, int i) {
    return (particles + PARTICLE_SIZE*i);
}

/*
 * Sums particle weights.
 * Needs to run in a single block.
 */
 __global__ void sum_weights(float *particles, double *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;

    for (int i = idx; i < N_PARTICLES; i += THREADS) {
        float *particle = get_particle(particles, i);
        sum += (double)particle[3];
    }

    __shared__ double r[THREADS];
    r[idx] = sum;
    __syncthreads();

    for (int size = THREADS/2; size > 0; size /= 2) {
        if (idx < size) {
            r[idx] += r[idx + size];
        }
        __syncthreads();
    }

    if (idx == 0) {
        *out = r[0];
    }
}

/*
 * Rescales particle weights so that \sum_i w_i = 1
 */
 __global__ void divide_weights(float *particles, double *s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = (float)s[0];
    float *particle = get_particle(particles, i);
    particle[3] /= sum;
}