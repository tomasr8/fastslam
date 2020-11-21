#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define PARTICLE_SIZE <<PARTICLE_SIZE>>

__device__ float* get_particle(float *particles, int i) {
    return (particles + PARTICLE_SIZE*i);
}

/*
 * Rescales particle weights such that \sum_i w_i = 1
 */
__global__ void rescale(float *particles, int n_particles) {
    float s = 0;

    for(int i = 0; i < n_particles; i++) {
        float *particle = get_particle(particles, i);
        s += particle[3];
    }

    // prevent float underflow
    s += 1.e-30;

    for(int i = 0; i < n_particles; i++) {
        float *particle = get_particle(particles, i);
        particle[3] /= s;
    }
}