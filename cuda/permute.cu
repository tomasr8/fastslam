#include <stdbool.h>

__global__ void reset(int *aux) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    aux[i] = -1;
}

__global__ void compute_positions(int *ancestors, int *aux) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    aux[ancestors[i]] = i;
}

__device__ bool needs_swap(int *ancestors, int *aux, int i) {
    int a = ancestors[i];
    return (a != i) && (aux[i] != -1);
}

__device__ bool is_end(int *ancestors, int *aux, int i) {
    int a = ancestors[i];
    return (ancestors[a] == a) || (aux[a] != i);
}

__device__ void swap(int *a, int *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

__global__ void permute(int *ancestors, int *aux, int block_size, int n) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    for(int k = 0; k < block_size; k++) {
        int i = thread_id*block_size + k;
    
        if(i >= n) {
            return;
        }

        bool end = is_end(ancestors, aux, i);
        __syncthreads();
    
        if(end) {
            int j = i;
            while(needs_swap(ancestors, aux, j)) {
                int pos = aux[j];
    
                if(j == pos) {
                    break;
                }
    
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
}