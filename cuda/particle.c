#include <stdio.h>
#include <stdbool.h>
#include "particle.h"

__device__ float* get_mean(float *particle, int i)
{
    return (particle + 6 + 2*i);
}

__device__ float* get_cov(float *particle, int i)
{
    int max_landmarks = (int)particle[4];
    return (particle + 6 + 2*max_landmarks + 4*i);
}

__device__ int get_n_landmarks(float *particle)
{
    return (int)particle[5];
}

__device__ void add_landmark(float *particle, float *mean, float *cov)
{
    particle[5]++;
    int n_landmarks = particle[5];
    float *new_mean = get_mean(particle, n_landmarks - 1);
    float *new_cov = get_cov(particle, n_landmarks - 1);

    new_mean[0] = mean[0];
    new_mean[1] = mean[1];

    new_cov[0] = cov[0];
    new_cov[1] = cov[1];
    new_cov[2] = cov[2];
    new_cov[3] = cov[3];
}

__device__ void add_unassigned_measurements_as_landmarks(float *particle, bool *assigned_measurements, float measurements[][2], int n_measurements, float *measurement_cov)
{
    for(int i = 0; i < n_measurements; i++) {
        if(!assigned_measurements[i]) {
            float x = particle[0];
            float y = particle[1];
            float measurement[] = { x + measurements[i][0], y + measurements[i][1] };

            add_landmark(particle, measurement, measurement_cov);
        }
    }
}
