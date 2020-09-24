#define __device__ 

__device__ float* get_mean(float *particle, int i);
__device__ float* get_cov(float *particle, int i);
__device__ int get_n_landmarks(float *particle);