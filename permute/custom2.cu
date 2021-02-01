#include <stdbool.h>

__global__ void reset(int *aux) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    aux[i] = -1;
}

__global__ void compute_positions(int *ancestors, int *aux) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // atomicMax(aux + ancestors[i], i);
    aux[ancestors[i]] = i;
}

__global__ void compute_end(int *ancestors, int *aux, int *end) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int a = ancestors[i];
    if((ancestors[a] == a) || (aux[a] != i)) {
        end[i] = 1;
    } else {
        end[i] = 0;
    }
}

__device__ bool needs_swap(int *ancestors, int *aux, int i) {
    int a = ancestors[i];
    return (a != i) && (aux[i] != -1);
}

__device__ void swap(int *a, int *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}



/**
 * Permutes the ancestor vector so that ancestors[i] = i if i is in ancestors.
 * Needs __syncthreads() so has to be run in a single block - can be circumvented by running the first part
 * in a separate kernel and saving the result in global/shared memory
 *
 * Requires N_PARTICLES to be a multiple of THREADS - can be fixed by adding if blocks before and after __syncthreads();
 */
__global__ void permute_custom(int *ancestors, int *aux, int *ends) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int end = ends[i];

    if(end == 1) {
        int j = i;
        while(needs_swap(ancestors, aux, j)) {
            int pos = aux[j];

            // if(j == pos) {
            //     break;
            // }

            if(aux[ancestors[j]] == j) {
                aux[ancestors[j]] = pos;
            }

            if(aux[ancestors[pos]] == pos) {
                aux[ancestors[pos]] = j;
            }

            swap(&ancestors[j], &ancestors[pos]);

            j = pos;
        }
    }
}