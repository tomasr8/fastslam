#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#include "sort.h"
#include "particle.h"

typedef struct 
{
    float *matrix;
    int n_landmarks;
    int n_measurements;
} dist_matrix;

typedef struct 
{
    int *assignment;
    bool *assigned_landmarks;
    bool *assigned_measurements;
} assignment;

float pdf(float *x, float *mean, float* cov);