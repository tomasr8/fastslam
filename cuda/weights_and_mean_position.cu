#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define PARTICLE_SIZE <<PARTICLE_SIZE>>

__device__ float* get_particle(float *particles, int i) {
    return (particles + PARTICLE_SIZE*i);
}

__global__ void get_weights_and_mean_position(float *particles, int n_particles, float *weights, float *mean) {
    float x = 0;
    float y = 0;
    float theta = 0;

    for(int i = 0; i < n_particles; i++) {
        float *particle = get_particle(particles, i);
        weights[i] = particle[3];
        x += particle[3] * particle[0];
        y += particle[3] * particle[1];
        theta += particle[3] * particle[2];
    }

    mean[0] = x;
    mean[1] = y;
    mean[2] = theta;
}