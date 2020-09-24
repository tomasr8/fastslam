#define __device__ 

__device__ float* get_mean(float *particle, int i);
__device__ float* get_cov(float *particle, int i);
__device__ int get_n_landmarks(float *particle);
__device__ void add_landmark(float *particle, float *mean, float *cov);
__device__ void add_unassigned_measurements_as_landmarks(float *particle, bool *assigned_measurements, float measurements[][2], int n_measurements, float *measurement_cov);