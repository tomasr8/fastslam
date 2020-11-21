#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <curand_kernel.h>

#ifndef M_PI
#define M_PI 3.14159265359
#endif

#define MIN(a,b) (((a)<(b))?(a):(b))

#define PARTICLE_SIZE <<PARTICLE_SIZE>>

__device__ float* get_particle(float *particles, int i) {
    return (particles + PARTICLE_SIZE*i);
}


extern "C" { // name mangling shenanigans
__device__ curandState_t* states[<<THREADS>>];

__global__ void init_rng(int seed)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t* s = new curandState_t;
    curand_init(seed, tidx, 0, s);
    states[tidx] = s;
}


__global__ void predict(float *particles, int block_size, int n_particles, float ua, float ub, float sigma_a, float sigma_b, float dt) {
    if(ua == 0.0 && ub == 0.0) {
        return;
    }

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int id_min = thread_id*block_size;
    int id_max = MIN(thread_id*block_size + (block_size - 1), n_particles - 1);

    for(int i = id_min; i <= id_max; i++) {
        float *particle = get_particle(particles, i);
        particle[2] += ua + sigma_a * curand_normal(states[thread_id]);
        particle[2] = fmod(particle[2], (float)(2*M_PI));

        float dist = (ub * dt) + sigma_b * curand_normal(states[thread_id]);
        particle[0] += cos(particle[2]) * dist;
        particle[1] += sin(particle[2]) * dist;
    }
}
}