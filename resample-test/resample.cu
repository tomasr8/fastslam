#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <curand_kernel.h>

#define MAX(a,b) (((a)>(b))?(a):(b))
#define THREADS 1024
#define B 10
#define N 8192

extern "C" {
__device__ curandState_t* states[THREADS];


// This function is only called once to initialize the rngs.
__global__ void init_rng(int seed)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t* s = new curandState_t;
    curand_init(seed, tidx, 0, s);
    states[tidx] = s;
}

__global__ void metropolis(float *weights, int n, int *indices) {
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    int idx = block_id * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    for (int i = idx; i < n; i += THREADS) {
        int p = idx;
        for(int j = 0; j < B; j++) {
            float u = curand_uniform(states[idx]);
            int q = (int)floor(curand_uniform(states[idx]) * N);
    
            if(u <= weights[q]/weights[p]) {
                p = q;
            }
        }
        indices[idx] = p;
    }
}

__global__ void rejection(float *weights, int n, float max, int *indices) {
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    int idx = block_id * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    for (int i = idx; i < n; i += THREADS) {
        int p = idx;
        float u = curand_uniform(states[idx]);

        while(u <= weights[p]/max) {
            p = (int)floor(curand_uniform(states[idx]) * N);
            u = curand_uniform(states[idx]);
        }

        indices[idx] = p;
    }
}


__global__ void systematic(float *weights, float *cumsum, int n, float rand, int *indices) {
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    int idx = block_id * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    
    int left;
    int right;
    for (int k = idx; k < n; k += THREADS) {
        left = ceil(((cumsum[k] - weights[k]) * n) - rand);
        right = ceil((cumsum[k] * n) - rand);
        for(int j = left; j < right; j++) {
            indices[j] = k;
        }
    }
}



__global__ void scan(float *g_odata, float *g_idata, int n){
    extern __shared__float temp[]; // allocated on invocation
    int thid = threadIdx.x;
    int pout = 0, pin = 1;
    // load input into shared memory.
    // Exclusive scan: shift right by one and set first element to 0
    temp[thid] = (thid > 0) ? g_idata[thid-1] : 0;
    
    __syncthreads();
    
    for( int offset = 1; offset < n; offset <<= 1 ){
        pout = 1 - pout;
        // swap double buffer indices
        pin  = 1 - pout;
        if(thid >= offset)
            temp[pout*n+thid] += temp[pin*n+thid - offset];
        else
            temp[pout*n+thid] = temp[pin*n+thid];
            
        __syncthreads();
    }
    
    g_odata[thid] = temp[pout*n+thid];
    // write output
}



__global__ void sum_weights(float *weights, int n, float *out) {
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    int idx = block_id * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    float sum = 0;

    for (int i = idx; i < n; i += THREADS) {
        sum = MAX(sum, weights[i]);
    }

    __shared__ float r[THREADS];
    r[idx] = sum;
    __syncthreads();

    for (int size = THREADS/2; size>0; size/=2) { //uniform
        if (idx<size) {
            r[idx] += r[idx+size];
        }
        __syncthreads();
    }

    if (idx == 0) {
        *out = r[0];
    }
}
}