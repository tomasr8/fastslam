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
__global__ void predict(float *particles, int block_size, int n_particles,
    float x, float y, float theta, float sigma_x, float sigma_y, float sigma_theta) {
    
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int id_min = thread_id*block_size;
    int id_max = MIN(thread_id*block_size + (block_size - 1), n_particles - 1);

    for(int i = id_min; i <= id_max; i++) {
        float *particle = get_particle(particles, i);
        // curand_normal() samples from standard normal
        // to get a general N(mu, sigma), we use Y = mu + sigma*X,
        // though in our case mu=0.
        particle[0] = x + sigma_x * curand_normal(states[thread_id]);
        particle[1] = y + sigma_y * curand_normal(states[thread_id]);
        particle[2] = theta + sigma_theta * curand_normal(states[thread_id]);
    }
}
}