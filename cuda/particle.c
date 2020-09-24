#include "particle.h"

__device__ float* get_mean(float *particle, int i)
{
    return (particle + 5 + 2*i);
}

__device__ float* get_cov(float *particle, int i)
{
    int n_landmarks = particle[4];
    return (particle + 5 + 2*n_landmarks + 4*i);
}

__device__ int get_n_landmarks(float *particle)
{
    return (int)particle[4];
}