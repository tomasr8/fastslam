#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define MIN(a,b) (((a)<(b))?(a):(b))

#define PARTICLE_SIZE <<PARTICLE_SIZE>>
#define BLOCK_SIZE <<BLOCK_SIZE>>

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

    // assumes n_particles is a multiple of block_size
    int start = thread_id*block_size;

    for(int i = 0; i < BLOCK_SIZE; i++) {
        float *old_particle = get_particle(old_particles, idx[start+i]);
        float *new_particle = get_particle(new_particles, start+i);

        int max_landmarks = (int)old_particle[4];
        int n_landmarks = (int)old_particle[5];

        new_particle[0] = old_particle[0];
        new_particle[1] = old_particle[1];
        new_particle[2] = old_particle[2];
        new_particle[3] = 1.0/n_particles;
        new_particle[4] = old_particle[4];
        new_particle[5] = old_particle[5];

        for(int k = 0; k < n_landmarks; k++) {
            new_particle[6+2*k] = old_particle[6+2*k];
            new_particle[6+2*k+1] = old_particle[6+2*k+1];

            new_particle[6+2*max_landmarks+4*k] = old_particle[6+2*max_landmarks+4*k];
            new_particle[6+2*max_landmarks+4*k+1] = old_particle[6+2*max_landmarks+4*k+1];
            new_particle[6+2*max_landmarks+4*k+2] = old_particle[6+2*max_landmarks+4*k+2];
            new_particle[6+2*max_landmarks+4*k+3] = old_particle[6+2*max_landmarks+4*k+3];

            new_particle[6+6*max_landmarks+k] = old_particle[6+6*max_landmarks+k];
        }
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