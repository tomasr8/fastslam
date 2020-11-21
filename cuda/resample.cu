#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define MIN(a,b) (((a)<(b))?(a):(b))

#define PARTICLE_SIZE <<PARTICLE_SIZE>>

__device__ float* get_particle(float *particles, int i) {
    return (particles + PARTICLE_SIZE*i);
}

/*
 * Copies particles from one memory block to another based on indices
 * given by systematic resampling.
 *
 * The systematic resampling is handled on the host. I haven't yet figured out
 * how to do it efficiently on the device and the CPU implementation is fast enough.
 */
__global__ void resample(
    float *old_particles, float *new_particles, int *idx, int block_size, int n_particles)
{

    // *idx is a mapping where i is the index of the new particle and
    // idx[i] is the index of the old particle.

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    // int particle_size = 6 + 7*((int)old_particles[4]);
    int id_min = thread_id*block_size;
    int id_max = thread_id*block_size + (block_size - 1);
    // int id_max = MIN(thread_id*block_size + (block_size - 1), n_particles - 1);


    for(int i = id_min; i <= id_max; i++) {
        float *old_particle = get_particle(old_particles, idx[i]);
        float *new_particle = get_particle(new_particles, i);

        for(int k = 0; k < PARTICLE_SIZE; k++) {
            new_particle[k] = old_particle[k];
        }

        new_particle[3] = 1.0/n_particles;
    }
}

// __global__ void resample(
//     float *particles, float *new_particles, int block_size, int n_particles, float random)
// {
//     // return;
//     int block_id = blockIdx.x+ blockIdx.y * gridDim.x;
//     int thread_id = block_id * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

//     // if(thread_id == 0) {
//     //     printf("dfgdfgd\n");
//     // }

//     int particle_size = 6 + 7*((int)particles[4]);
//     int id_min = thread_id*block_size;
//     int id_max = MIN(thread_id*block_size + (block_size - 1), n_particles - 1);


//     int size = (n_particles) * sizeof(float);
//     float *cumsum;

//     cumsum = (float *)malloc(size);
//     cumsum[0] = get_particle(particles, 0)[3];

//     for(int i = 1; i < n_particles; i++) {
//         cumsum[i] = cumsum[i-1] + get_particle(particles, i)[3];
//     }

//     cumsum[n_particles-1] = 1.0;

//     int i = 0;
//     int j = 0;
//     while(i < n_particles) {
//         if( ((i + random)/n_particles) < cumsum[j] ) {

//             if(i >= id_min && i <= id_max) {
//                 float *new_particle = get_particle(new_particles, i);
//                 float *old_particle = get_particle(particles, j);

//                 // memcpy(new_particle, old_particle, particle_size);

//             }

//             if(i > id_max) {
//                 break;
//             }

//             i += 1;
//         } else {
//             j += 1;
//         }
//     }

//     free(cumsum);
// }