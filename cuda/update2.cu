#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

// #define M_PI 3.14159265359
// particle

__device__ float* get_particle(float *particles, int i) {
    int max_landmarks = (int)particles[4];
    return (particles + (6 + 6*max_landmarks)*i);
}

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

__device__ void add_landmark(float *particle, float mean[2], float *cov)
{
    // printf("add_landmark %f %f\n", mean[0], mean[1]);
    // if(particle[5] == particle[4]) {
    //     printf(">>>>>>>>>>>>>>>>>> TOO MANY LANDMARKS\n");
    // }

    int n_landmarks = (int)particle[5];

    // particle[5] = (float)(n_landmarks + 1);
    // printf("max_l: %f\n", particle[5]);
    particle[6] = 70.0;
    particle[7] = 70.0;
    particle[5] = 70.0;



    // float *new_mean = get_mean(particle, n_landmarks - 1);
    // float *new_cov = get_cov(particle, n_landmarks - 1);

    // new_mean[0] = mean[0];
    // new_mean[1] = mean[1];

    // new_cov[0] = cov[0];
    // new_cov[1] = cov[1];
    // new_cov[2] = cov[2];
    // new_cov[3] = cov[3];
}

__device__ void add_unassigned_measurements_as_landmarks(float *particle, bool *assigned_measurements, float measurements[][2], int n_measurements, float *measurement_cov)
{
    for(int i = 0; i < n_measurements; i++) {
        if(!assigned_measurements[i]) {
            float x = particle[0];
            float y = particle[1];
            float measurement[] = { x + measurements[i][0], y + measurements[i][1] };
            
            // printf("add_landmark %f %f %f %f\n", x, y, measurements[i][0], measurements[i][1]);

            // add_landmark(particle, measurement, measurement_cov);
            // particle[5] = 70.0;
        }
    }
}

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


__device__ void compute_dist_matrix(dist_matrix *matrix)
{
    for(int i = 0; i < matrix->n_landmarks; i++) {
        for(int j = 0; j < matrix->n_measurements; j++) {
            matrix->matrix[i * matrix->n_measurements + j] = 0.0;
        }
    }
}

__device__ void associate_landmarks_measurements(int n_landmarks, int n_measurements) {
    if(n_landmarks > 0 && n_measurements > 0) {
        dist_matrix *matrix = (dist_matrix *)malloc(sizeof(dist_matrix));
        if(matrix == NULL) {
            printf(">>>>>>>>>>>>>>>>>>>>>>>> MALLOC FAILED\n");
        }


        matrix->matrix = (float *)malloc(n_landmarks * n_measurements * sizeof(float));
        if(matrix->matrix == NULL) {
            printf("size %d \n", n_landmarks * n_measurements);
            printf(">>>>>>>>>>>>>>>>>>>>>>>> MALLOC FAILED 2\n");
        }
        matrix->n_landmarks = n_landmarks;
        matrix->n_measurements = n_measurements;

        compute_dist_matrix(matrix);

        free(matrix->matrix);
        free(matrix);
    }
}


__global__ void update(float *particles, float measurements[][2], int n_particles, int n_measurements, float *measurement_cov, float threshold)
{
    // int i = threadIdx.x + blockIdx.x * blockDim.x;

    int blockId = blockIdx.x+ blockIdx.y * gridDim.x;
    int i = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if(i >= n_particles) {
        return;
    }

    float *particle = get_particle(particles, i);
    int n_landmarks = get_n_landmarks(particle);
    if(n_landmarks > 0 && n_measurements > 0) {

        bool *assigned_landmarks = (bool *)malloc(n_landmarks * sizeof(bool));
        bool *assigned_measurements = (bool *)malloc(n_measurements * sizeof(bool));

        if(assigned_landmarks == NULL) {
            printf(">>>>>>>>>>>>>>>>>>>>>>>> MALLOC FAILED x1\n");
        }

        if(assigned_measurements == NULL) {
            printf(">>>>>>>>>>>>>>>>>>>>>>>> MALLOC FAILED x2\n");
        }

        for(int i = 0; i < n_landmarks; i++) {
            assigned_landmarks[i] = false;
        }

        for(int i = 0; i < n_measurements; i++) {
            assigned_measurements[i] = false;
        }

        int *assignment_lm = (int *)malloc(n_landmarks * sizeof(int));
        for(int i = 0; i < n_landmarks; i++) {
            assignment_lm[i] = -1;
        }

        if(assignment_lm == NULL) {
            printf(">>>>>>>>>>>>>>>>>>>>>>>> MALLOC FAILED x3\n");
        }

        assignment *assignmentx = (assignment*)malloc(sizeof(assignment));

        if(assignmentx == NULL) {
            printf(">>>>>>>>>>>>>>>>>>>>>>>> MALLOC FAILED x4\n");
        }

        assignmentx->assignment = assignment_lm;
        assignmentx->assigned_landmarks = assigned_landmarks;
        assignmentx->assigned_measurements = assigned_measurements;

        int max_landmarks = (int)particles[4];
        float a = 70.0;
        particles[(6 + 6*max_landmarks)*i + 5] = a;

        associate_landmarks_measurements(
            n_landmarks, n_measurements
        );

        // update_landmark(particle, measurements, assignmentx, n_measurements, measurement_cov);

        // add_unassigned_measurements_as_landmarks(particle, assignmentx->assigned_measurements, measurements, n_measurements, measurement_cov);
        // int max_landmarks = (int)particles[4];
        // float a = 70.0;
        // particles[(6 + 6*max_landmarks)*i + 5] = a;

        // float a = 70.0;
        // particle[5] = a;

        free(assignmentx->assigned_landmarks);
        free(assignmentx->assigned_measurements);
        free(assignmentx->assignment);
        free(assignmentx);
    }
}