import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

cuda_update = SourceModule("""
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define M_PI 3.14159265359
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

// =================================================
// =================================================
// =================================================
// sort

void swap(float *a, float *b)
{
    float t = *a;
    *a = *b;
    *b = t;
}

void swap_idx(int *a, int *b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

int partition(float arr[], int lm[], int me[], int low, int high)
{
    float pivot = arr[high]; // pivot
    int i = (low - 1);       // Index of smaller element

    for (int j = low; j <= high - 1; j++)
    {
        // If current element is smaller than the pivot
        if (arr[j] > pivot)
        {
            i++; // increment index of smaller element
            swap(&arr[i], &arr[j]);
            swap_idx(&lm[i], &lm[j]);
            swap_idx(&me[i], &me[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    swap_idx(&lm[i + 1], &lm[high]);
    swap_idx(&me[i + 1], &me[high]);
    return (i + 1);
}


void quicksort(float arr[], int lm[], int me[], int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now 
           at right place */
        int pi = partition(arr, lm, me, low, high);

        // Separately sort elements before
        // partition and after partition
        quicksort(arr, lm, me, low, pi - 1);
        quicksort(arr, lm, me, pi + 1, high);
    }
}

// =================================================
// =================================================
// =================================================
// data assoc

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

float pdf(float *x, float *mean, float* cov)
{
    float a = cov[0];
    float b = cov[1];

    float logdet = log(a*a - b*b);

    float root = sqrt(2)/2.0;
    float e = root * (1/sqrt(a-b));
    float f = root * (1/sqrt(a+b));

    float m = x[0] - mean[0];
    float n = x[1] - mean[1];

    float maha = 2*(m*m*e*e + n*n*f*f);
    float log2pi = log(2 * M_PI);
    return exp(-0.5 * (2*log2pi + maha + logdet));
}

void compute_dist_matrix(float **landmarks, float **measurements, dist_matrix *matrix, float *landmarks_cov, float *measurement_cov)
{
    for(int i = 0; i < matrix->n_landmarks; i++) {
        for(int j = 0; j < matrix->n_measurements; j++) {
            float cov[4] = { landmarks_cov[i] + measurement_cov[0], landmarks_cov[i+1], landmarks_cov[i+2], landmarks_cov[i+3] + measurement_cov[1] };
            matrix->matrix[i * matrix->n_measurements + j] = pdf(landmarks[i], measurements[j], cov);
        }
    }
}

void assign(dist_matrix *matrix, assignment *assignment, float threshold) {
    int n_landmarks = matrix->n_landmarks;
    int n_measurements = matrix->n_measurements;

    int *landmark_idx = malloc(n_landmarks * n_measurements * sizeof(int));
    int *measurement_idx = malloc(n_landmarks * n_measurements * sizeof(int));

    for(int i = 0; i < n_landmarks; i++) {
        for(int j = 0; j < n_measurements; j++) {
            landmark_idx[i * n_measurements + j] = i;
            measurement_idx[i * n_measurements + j] = j;
        }
    }

    // float *dist_matrix = malloc(n_landmarks * n_measurements * sizeof(float));
    float *matrix_copy = malloc(n_landmarks * n_measurements * sizeof(float));
    for (int i = 0; i < n_landmarks * n_measurements; i++) {
        matrix_copy[i] = matrix->matrix[i]; 
    }

    quicksort(matrix_copy, landmark_idx, measurement_idx, 0, (n_landmarks * n_measurements) - 1);

    free(matrix_copy);

    int assigned_total = 0;
    float cost = 0;

    for(int i = 0; i < n_landmarks * n_measurements; i++) {
        int a = landmark_idx[i];
        int b = measurement_idx[i];

        if(assignment->assigned_landmarks[a] || assignment->assigned_measurements[b]){
            continue;
        }
        
        if(matrix->matrix[a * n_measurements + b] > threshold){
            assignment->assignment[a] = b;
            assignment->assigned_landmarks[a] = true;
            assignment->assigned_measurements[b] = true;
            assigned_total += 1;
            cost += matrix->matrix[a * n_measurements + b];
        }

        if(assigned_total == n_landmarks) {
            break;
        }
    }

    printf("Cost: %f\n", cost);

    free(landmark_idx);
    free(measurement_idx);
}

void associate_landmarks_measurements(float *particle, float measurements[][2], int n_landmarks, int n_measurements, assignment *assignment, float *measurement_cov, float threshold) {
    float pos[] = { particle[0], particle[1] };
    float **measurement_predicted = malloc(n_landmarks * sizeof(float*));

    for(int i = 0; i < n_landmarks; i++) {
        measurement_predicted[i] = malloc(2 * sizeof(float));
        float *landmark = get_mean(particle, i);
        measurement_predicted[i][0] = landmark[0] - pos[0];
        measurement_predicted[i][1] = landmark[1] - pos[1];
    }

    float *landmarks_cov = get_cov(particle, n_landmarks);

    dist_matrix *matrix = malloc(sizeof(dist_matrix));
    matrix->matrix = malloc(n_landmarks * n_measurements * sizeof(float));;
    matrix->n_landmarks = n_landmarks;
    matrix->n_measurements = n_measurements;

    compute_dist_matrix(measurement_predicted, measurements, matrix, landmarks_cov, measurement_cov);

    assign(matrix, assignment, threshold);
    add_unassigned_measurements_as_landmarks(particle, assignment->assigned_measurements, measurements, n_measurements, measurement_cov);

    for(int i = 0; i < n_landmarks; i++) {
        free(measurement_predicted[i]);
    }
    free(measurement_predicted);

    free(matrix->matrix);
    free(matrix);
    // free(assigned_landmarks);
    // free(assigned_measurements);
    // free(assignment_lm);
    // free(assignment);
}

__device__ update_landmark(float *particle, float measurements[][2], assignment *assignment, int n_measurements, float *measurement_cov)
{

    float x = particle[0];
    float y = particle[1];
    int n_landmarks = get_n_landmarks(particle);

    for(int i = 0; i < n_landmarks; i++) {
        int j = assignment->assignment[i];

        if(j == -1) {
            continue;
        }

        float *mean = get_mean(particle, i);
        float mean_x = mean[0];
        float mean_y = mean[1];

        float measurement_predicted[2] = { mean_x - x, mean_y - y };
        float residual[2] = { measurements[j][0] - measurement_predicted[0], measurements[j][1] - measurement_predicted[1] };

        float *cov = get_covariance(particle, i);

        float Q[4] = { cov[0] + measurement_cov[0], cov[1], cov[2], cov[3] + measurement_cov[1] };
        float K[4] = { 0, 0, 0, 0 };
        float Q_inv[4] = { 0, 0, 0, 0};
        pinv(Q, Q_inv);
        matmul(cov, Q_inv, K);

        particle[3] *= pdf(measurements[j], measurement_predicted, Q);
    }
}

__global__ void update(float *particles, float measurements[][2], int n_particles, int n_measurements, float *measurement_cov, float threshold)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i >= n_particles) {
        return;
    }

    float *particle = get_particle(particles, i);
    int n_landmarks = get_n_landmarks(particle);

    bool *assigned_landmarks = malloc(n_landmarks * sizeof(bool));
    bool *assigned_measurements = malloc(n_measurements * sizeof(bool));

    for(int i = 0; i < n_landmarks; i++) {
        assigned_landmarks[i] = false;
    }

    for(int i = 0; i < n_measurements; i++) {
        assigned_measurements[i] = false;
    }

    int *assignment_lm = malloc(n_landmarks * sizeof(int));
    for(int i = 0; i < n_landmarks; i++) {
        assignment_lm[i] = -1;
    }

    assignment *assignment = malloc(sizeof(assignment));
    assignment->assignment = assignment_lm;
    assignment->assigned_landmarks = assigned_landmarks;
    assignment->assigned_measurements = assigned_measurements;

    associate_landmarks_measurements(
        particle, measurements,
        n_landmarks, n_measurements, assignment,
        measurement_cov, threshold
    );

    update_landmark(particle, measurements, assignment, n_measurements, measurement_cov);

    free(assignment->assigned_landmarks);
    free(assignment->assigned_measurements);
    free(assignment->assignment);
    free(assignment);
}

""")