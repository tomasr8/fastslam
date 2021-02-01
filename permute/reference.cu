#include <stdbool.h>

__global__ void reset(int *d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d[i] = n;
}

__global__ void prepermute(int *ancestors, int *d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    d[ancestors[i]] = i;

    // int *p = d + ancestors[i];
    // atomicMin(p, i);
}


// __global__ void permute_reference(int *ancestors, int *c, int *d, int block_size, int n) {
//     int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

//     for(int k = 0; k < block_size; k++) {
//         int i = thread_id*block_size + k;

//         int x = d[ancestors[i]];
//         if(x != i) {
//             x = i;
//             while(d[x] < n) {
//                 x = d[x];
//             }
//             d[x] = i;
//         }
    
//     }

//     __syncthreads();

//     for(int k = 0; k < block_size; k++) {
//         int i = thread_id*block_size + k;

//         c[i] = ancestors[d[i]];
//     }
// }


__global__ void permute_reference(int *ancestors, int *c, int *d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int x = d[ancestors[i]];
    if(x != i) {
        x = i;
        while(d[x] < n) {
            x = d[x];
        }
        d[x] = i;
    }
}

__global__ void write_to_c(int *ancestors, int *c, int *d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    c[i] = ancestors[d[i]];
}