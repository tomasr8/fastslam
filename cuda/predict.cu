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

// Manual extern "C" to stop name mangling shenanigans
// Otherwise doesn't compile because curand complains
extern "C" {

// Based on https://stackoverflow.com/questions/46169633/how-to-generate-random-number-inside-pycuda-kernel    
// Each thread has a random state
__device__ curandState_t* states[<<THREADS>>];


// This function is only called once to initialize the rngs.
__global__ void init_rng(int seed)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t* s = new curandState_t;
    curand_init(seed, tidx, 0, s);
    states[tidx] = s;
}

// Moves particles based on the control input and movement model.
// In the future, this will probably be replaced to reflect the
// actual sensors in the car e.g. IMU. 
__global__ void predict(float *particles, int block_size, int n_particles, float ua, float ub, float sigma_a, float sigma_b, float dt) {
    if(ua == 0.0 && ub == 0.0) {
        return;
    }

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int id_min = thread_id*block_size;
    int id_max = MIN(thread_id*block_size + (block_size - 1), n_particles - 1);

    for(int i = id_min; i <= id_max; i++) {
        float *particle = get_particle(particles, i);
        // curand_normal() samples from standard normal
        // to get a general N(mu, sigma), we use Y = mu + sigma*X,
        // though in our case mu=0.
        particle[2] += ua + sigma_a * curand_normal(states[thread_id]);
        particle[2] = fmod(particle[2], (float)(2*M_PI));

        float dist = (ub * dt) + sigma_b * curand_normal(states[thread_id]);
        particle[0] += cos(particle[2]) * dist;
        particle[1] += sin(particle[2]) * dist;
    }
}
}