#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#include "sort.h"

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

    // for (int i = 0; i < n_landmarks * n_measurements; i++) {
    //     printf("%f, ", matrix_copy[i]); 
    // }
    // printf("\n");

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


void compute_dist_matrix(float **landmarks, float **measurements, dist_matrix *matrix, float *landmarks_cov, float *measurement_cov)
{
    for(int i = 0; i < matrix->n_landmarks; i++) {
        for(int j = 0; j < matrix->n_measurements; j++) {
            float cov[4] = { landmarks_cov[i] + measurement_cov[0], landmarks_cov[i+1], landmarks_cov[i+2], landmarks_cov[i+3] + measurement_cov[1] };
            matrix->matrix[i * matrix->n_measurements + j] = pdf(landmarks[i], measurements[j], cov);
        }
    }
}


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


int main()
{
    srand(0);

    int n_landmarks = 1000;
    int n_measurements = 100;

    // float m[] = { 1, 2, 10, 3, 6, 22, 4, 6, 5, 6, 3, 2, 1, 99, 2, 2, 7, 8, 1, 6, 33, 44, 17, 11, 31 };

    float *m = malloc(n_landmarks * n_measurements * sizeof(float));
    for(int i = 0; i < n_landmarks * n_measurements; i++) {
        m[i] = (float)rand()/(float)(RAND_MAX);
    }


    dist_matrix *matrix = malloc(sizeof(dist_matrix));
    matrix->matrix = m;
    matrix->n_landmarks = n_landmarks;
    matrix->n_measurements = n_measurements;

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

    assign(matrix, assignment, 0.0);

    // for(int i = 0; i < n_landmarks; i++) {
    //     printf("%d -> %d\n", i, assignment->assignment[i]);
    // }

    // for(int i = 0; i < n_landmarks; i++) {
    //     printf("%d,", assigned_landmarks[i]);
    // }
    // printf("\n");

    // for(int i = 0; i < n_measurements; i++) {
    //     printf("%d,", assigned_measurements[i]);
    // }
    // printf("\n");


    // free(matrix->matrix);
    free(matrix);
    free(assigned_landmarks);
    free(assigned_measurements);
    free(assignment_lm);
    free(assignment);




    // int n = sizeof(arr) / sizeof(arr[0]);
    // quickSort(arr, lm, me, 0, n - 1);
    // printf("Sorted array: \n");
    // printArray(arr, lm, me, n);
    return 0;
}