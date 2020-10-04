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

            add_landmark(particle, measurement, measurement_cov);
        }
    }
}

__device__ void add_measurements_as_landmarks(float *particle, float measurements[][2], int n_measurements, float *measurement_cov)
{
    for(int i = 0; i < n_measurements; i++) {
        float x = particle[0];
        float y = particle[1];
        float measurement[] = { x + measurements[i][0], y + measurements[i][1] };

        add_landmark(particle, measurement, measurement_cov);
    }
}

// =================================================
// =================================================
// =================================================
// sort

__device__ void swap(float* a, float* b) 
{ 
    int t = *a; 
    *a = *b; 
    *b = t; 
}

__device__ void swap_idx(int *a, int *b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

__device__ int partition(float arr[], int lm[], int me[], int l, int h)
{ 
    int x = arr[h]; 
    int i = (l - 1); 

    for (int j = l; j <= h - 1; j++) { 
        if (arr[j] > x) {
            i++; 
            swap(&arr[i], &arr[j]);
            swap_idx(&lm[i], &lm[j]);
            swap_idx(&me[i], &me[j]);
        } 
    } 
    swap(&arr[i + 1], &arr[h]);
    swap_idx(&lm[i + 1], &lm[h]);
    swap_idx(&me[i + 1], &me[h]);
    return (i + 1);
} 

__device__ void quicksort(float arr[], int lm[], int me[], int l, int h) 
{ 
    int *stack = (int*)malloc((h-l+1) * sizeof(int));
    // int stack[h - l + 1]; 

    // initialize top of stack 
    int top = -1; 

    // push initial values of l and h to stack 
    stack[++top] = l; 
    stack[++top] = h; 

    // Keep popping from stack while is not empty 
    while (top >= 0) { 
        // Pop h and l 
        h = stack[top--]; 
        l = stack[top--]; 

        // Set pivot element at its correct position 
        // in sorted array 
        int p = partition(arr, lm, me, l, h); 

        // If there are elements on left side of pivot, 
        // then push left side to stack 
        if (p - 1 > l) { 
            stack[++top] = l; 
            stack[++top] = p - 1; 
        } 

        // If there are elements on right side of pivot, 
        // then push right side to stack 
        if (p + 1 < h) { 
            stack[++top] = p + 1; 
            stack[++top] = h; 
        } 
    }

    free(stack);
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

__device__ void vecmul(float *A, float *u, float *v)
{
    float a = A[0];
    float b = A[1];
    float c = A[2];
    float d = A[3];

    float e = u[0];;
    float f = v[1];;

    v[0] = a*e + b*f;
    v[1] = c*e + d*f;
}

__device__ void matmul(float *A, float *B, float *C)
{
    float a = A[0];
    float b = A[1];
    float c = A[2];
    float d = A[3];

    float e = B[0];;
    float f = B[1];;
    float g = B[2];;
    float h = B[3];;

    C[0] = a*e + b*g;
    C[1] = a*f + b*h;
    C[2] = c*e + d*g;
    C[3] = c*f + d*h;
}

__device__ void pinv(float *A, float *B)
{
    float a = A[0];
    float b = A[1];
    float c = A[2];
    float d = A[3];

    float e = a*a + c*c;
    float f = a*b + c*d;
    float g = a*b + c*d;
    float h = b*b + d*d;

    float scalar = 1/(e*h - f*g);
    float e_i = scalar * h;
    float f_i = scalar * (-f);
    float g_i = scalar * (-g);
    float h_i = scalar * e;

    B[0] = e_i*a + f_i*b;
    B[1] = e_i*c + f_i*d;
    B[2] = g_i*a + h_i*b;
    B[3] = g_i*c + h_i*d;
}

__device__ float pdf(float *x, float *mean, float* cov)
{
    float a = cov[0];
    float b = cov[1];

    float logdet = log(a*a - b*b);

    float root = sqrt(2.0)/2.0;
    float e = root * (1.0/sqrt(a-b));
    float f = root * (1.0/sqrt(a+b));

    float m = x[0] - mean[0];
    float n = x[1] - mean[1];

    float maha = 2*(m*m*e*e + n*n*f*f);
    float log2pi = log(2 * M_PI);
    return exp(-0.5 * (2*log2pi + maha + logdet));
}

__device__ void compute_dist_matrix(float *landmarks, float measurements[][2], dist_matrix *matrix, float *landmarks_cov, float *measurement_cov)
{
    // printf("---\n");
    // printf("m_cov: [%f %f %f %f]\n", measurement_cov[0], measurement_cov[1], measurement_cov[2], measurement_cov[3]);
    for(int i = 0; i < matrix->n_landmarks; i++) {
        // printf("lm_cov: [%f %f %f %f]\n", landmarks_cov[4*i], landmarks_cov[4*i+1], landmarks_cov[4*i+2], landmarks_cov[4*i+3]);

        for(int j = 0; j < matrix->n_measurements; j++) {


            float cov[4] = {
                landmarks_cov[4*i] + measurement_cov[0],
                landmarks_cov[4*i+1] + measurement_cov[1],
                landmarks_cov[4*i+2] + measurement_cov[2],
                landmarks_cov[4*i+3] + measurement_cov[3]
            };
            float landmark[] = { landmarks[2*i], landmarks[2*i + 1] };
            matrix->matrix[i * matrix->n_measurements + j] = pdf(landmark, measurements[j], cov);
            // printf("%.15f [%f %f], [%f %f], [%f %f %f %f]\n", pdf(landmarks[i], measurements[j], cov),
                // landmarks[i][0], landmarks[i][1], measurements[j][0], measurements[j][1], cov[0], cov[1], cov[2], cov[3]);
        }
    }
    // printf("---\n");

}

__device__ void assign(dist_matrix *matrix, assignment *assignment, float threshold) {
    int n_landmarks = matrix->n_landmarks;
    int n_measurements = matrix->n_measurements;

    int *landmark_idx = (int *)malloc(n_landmarks * n_measurements * sizeof(int));
    int *measurement_idx = (int *)malloc(n_landmarks * n_measurements * sizeof(int));

    for(int i = 0; i < n_landmarks; i++) {
        for(int j = 0; j < n_measurements; j++) {
            landmark_idx[i * n_measurements + j] = i;
            measurement_idx[i * n_measurements + j] = j;
        }
    }

    // float *dist_matrix = malloc(n_landmarks * n_measurements * sizeof(float));
    float *matrix_copy = (float *)malloc(n_landmarks * n_measurements * sizeof(float));
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

    // printf("Cost: %f\n", cost);

    free(landmark_idx);
    free(measurement_idx);
}

__device__ void associate_landmarks_measurements(float *particle, float measurements[][2], int n_landmarks, int n_measurements, assignment *assignment, float *measurement_cov, float threshold) {
    if(n_landmarks > 0 && n_measurements > 0) {
        float pos[] = { particle[0], particle[1] };
        float *measurement_predicted = (float *)malloc(2 * n_landmarks * sizeof(float));
        if(measurement_predicted == NULL) {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!============= MALLOC FAILED mp\n");
        }

        for(int i = 0; i < n_landmarks; i++) {
            // measurement_predicted[i] = (float *)malloc(2 * sizeof(float));
            float *landmark = get_mean(particle, i);
            measurement_predicted[2*i] = landmark[0] - pos[0];
            measurement_predicted[2*i + 1] = landmark[1] - pos[1];
        }

        float *landmarks_cov = get_cov(particle, 0);

        dist_matrix *matrix = (dist_matrix *)malloc(sizeof(dist_matrix));
        if(matrix == NULL) {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!============= MALLOC FAILED matrix\n");
        }
        matrix->matrix = (float *)malloc(n_landmarks * n_measurements * sizeof(float));
        if(matrix->matrix == NULL) {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!============= MALLOC FAILED matrix->matrix %d\n", n_landmarks * n_measurements);
        }
        matrix->n_landmarks = n_landmarks;
        matrix->n_measurements = n_measurements;

        compute_dist_matrix(measurement_predicted, measurements, matrix, landmarks_cov, measurement_cov);

        // assign(matrix, assignment, threshold);

        // for(int i = 0; i < n_landmarks; i++) {
        //     free(measurement_predicted[i]);
        // }
        free(measurement_predicted);

        free(matrix->matrix);
        free(matrix);
    }
}

__device__ void update_landmark(float *particle, float measurements[][2], assignment *assignment, int n_measurements, float *measurement_cov)
{
    float x = particle[0];
    float y = particle[1];
    int n_landmarks = get_n_landmarks(particle);
    // printf("in update landmark %d\n", n_landmarks);

    for(int i = 0; i < n_landmarks; i++) {
        // printf("in update lm loop %d\n", i);
        int j = assignment->assignment[i];
        // printf("%d -> %d\n", i, j);

        if(j == -1) {
            continue;
        }

        // printf("here-2\n");

        float *mean = get_mean(particle, i);
        float mean_x = mean[0];
        float mean_y = mean[1];

        // printf("here-1\n");

        float measurement_predicted[2] = { mean_x - x, mean_y - y };
        // printf("here\n");
        float residual[2] = { measurements[j][0] - measurement_predicted[0], measurements[j][1] - measurement_predicted[1] };

        // printf("residual: [%f %f]\n", residual[0], residual[1]);

        // printf("here0\n");
        float *cov = get_cov(particle, i);

        float Q[4] = {
            cov[0] + measurement_cov[0],
            cov[1] + measurement_cov[1],
            cov[2] + measurement_cov[2],
            cov[3] + measurement_cov[3]
        };

        // printf("Q: [%f %f %f %f]\n", Q[0], Q[1], Q[2], Q[3]);


        float K[4] = { 0, 0, 0, 0 };
        float Q_inv[4] = { 0, 0, 0, 0 };
        pinv(Q, Q_inv);
        matmul(cov, Q_inv, K);

        // printf("K: [%f %f %f %f]\n", K[0], K[1], K[2], K[3]);

        float K_residual[] = { 0, 0 };
        vecmul(K, residual, K_residual);
        mean[0] += K_residual[0];
        mean[1] += K_residual[1];

        // printf("here2\n");


        float new_cov[] = { 1 - K[0], K[1], K[2], 1 - K[3] };
        matmul(new_cov, cov, new_cov);
        cov[0] = new_cov[0];
        cov[1] = new_cov[1];
        cov[2] = new_cov[2];
        cov[3] = new_cov[3];

        // printf("here3\n");

        particle[3] *= pdf(measurements[j], measurement_predicted, Q);
        // printf("%f %f %f %f %f\n", measurements[j][0], measurements[j][1], measurement_predicted[0], measurement_predicted[1], pdf(measurements[j], measurement_predicted, Q));
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

    if(n_measurements == 0) {
        return;
    } else if(n_landmarks == 0 && n_measurements > 0) {
        add_measurements_as_landmarks(particle, measurements, n_measurements, measurement_cov);
        return;
    }

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

    associate_landmarks_measurements(
        particle, measurements,
        n_landmarks, n_measurements, assignmentx,
        measurement_cov, threshold
    );

    // update_landmark(particle, measurements, assignmentx, n_measurements, measurement_cov);

    // add_unassigned_measurements_as_landmarks(particle, assignmentx->assigned_measurements, measurements, n_measurements, measurement_cov);

    free(assignmentx->assigned_landmarks);
    free(assignmentx->assigned_measurements);
    free(assignmentx->assignment);
    free(assignmentx);

}