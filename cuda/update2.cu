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

__device__ void add_landmark(float *particle, float mean[2], float *cov)
{
    int n_landmarks = (int)particle[5];
    particle[5] = (float)(n_landmarks + 1);

    float *new_mean = get_mean(particle, n_landmarks);
    float *new_cov = get_cov(particle, n_landmarks);

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

__device__ void insertionSort(float arr[], int lm[], int me[], int n) 
{ 
    int i, lm_key, me_key, j;
    float key;
    for (i = 1; i < n; i++) { 
        key = arr[i];
        lm_key = lm[i];
        me_key = me[i];
        j = i - 1; 
  
        while (j >= 0 && arr[j] < key) { 
            arr[j + 1] = arr[j];
            lm[j + 1] = lm[j];
            me[j + 1] = me[j];
            j = j - 1; 
        } 
        arr[j + 1] = key;
        lm[j + 1] = lm_key;
        me[j + 1] = me_key; 
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
    bool *assigned_measurements;
} assignment;

typedef struct 
{
    int n_measurements;
    float *measurement_cov;
    float measurements[][2];
} landmark_measurements;


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

__device__ void compute_dist_matrix(float *particle, float measurements[][2], dist_matrix *matrix, float *measurement_cov)
{
    float pos[] = { particle[0], particle[1] };
    float *landmarks_cov = get_cov(particle, 0);

    for(int i = 0; i < matrix->n_landmarks; i++) {
        float *landmark = get_mean(particle, i);

        for(int j = 0; j < matrix->n_measurements; j++) {
            float measurement_predicted[] = {
                landmark[0] - pos[0], landmark[1] - pos[1]
            };

            float cov[4] = {
                landmarks_cov[4*i] + measurement_cov[0],
                landmarks_cov[4*i+1] + measurement_cov[1],
                landmarks_cov[4*i+2] + measurement_cov[2],
                landmarks_cov[4*i+3] + measurement_cov[3]
            };
            // float landmark[] = { landmarks[2*i], landmarks[2*i + 1] };
            matrix->matrix[i * matrix->n_measurements + j] = pdf(measurement_predicted, measurements[j], cov);
        }
    }
}

__device__ void assign(int i, dist_matrix *matrix, int *data_assoc_memory, assignment *assignment, float threshold) {
    int n_landmarks = matrix->n_landmarks;
    int n_measurements = matrix->n_measurements;

    // int usable = 0;
    // for (int i = 0; i < n_landmarks * n_measurements; i++) {
    //     if(matrix->matrix[i] > threshold) {
    //         usable++;
    //     }
    // }

    int *landmark_idx = data_assoc_memory;
    int *measurement_idx = landmark_idx + (n_landmarks * n_measurements);
    float *matrix_copy = (float *)(measurement_idx + (n_landmarks * n_measurements));

    for(int i = 0; i < n_landmarks; i++) {
        for(int j = 0; j < n_measurements; j++) {
            landmark_idx[i * n_measurements + j] = i;
            measurement_idx[i * n_measurements + j] = j;
        }
    }

    // if(i == 0) {
    //     for (int i = 0; i < n_landmarks * n_measurements; i++) {
    //         printf("%f\n", matrix->matrix[i]);
    //     }
    //     printf("=====\n");
    // }
    
    for (int i = 0; i < n_landmarks * n_measurements; i++) {
        matrix_copy[i] = matrix->matrix[i]; 
    }
    // printf("Usable: %d/%d\n", usable, (n_landmarks * n_measurements));
    if(i == 0) {
        printf("===BEFORE=== %d %d\n", n_landmarks, n_measurements);
        for (int i = 0; i < n_landmarks * n_measurements; i++) {
            printf("%f\n", matrix_copy[i]);
        }
        printf("=====\n");
    }

    // insertionSort(matrix_copy, landmark_idx, measurement_idx, (n_landmarks * n_measurements));
    insertionSort(matrix->matrix, landmark_idx, measurement_idx, (n_landmarks * n_measurements));

    if(i == 0) {
        printf("===AFTER===\n");
        for (int i = 0; i < n_landmarks * n_measurements; i++) {
            printf("%f\n", matrix_copy[i]);
        }
        printf("=====\n");
    }

    int assigned_total = 0;
    float cost = 0;

    for(int i = 0; i < n_landmarks * n_measurements; i++) {
        int a = landmark_idx[i];
        int b = measurement_idx[i];

        if(matrix->matrix[i] < threshold) {
            break;
        }

        if(assignment->assignment[a] != -1){
            continue;
        }

        assignment->assignment[a] = b;
        assignment->assigned_measurements[b] = true;
        assigned_total += 1;
        cost += matrix->matrix[i];

        if(assigned_total == n_landmarks) {
            break;
        }
    }

    if(i == 0) {
        printf("Cost: %f\n", cost);
        for(int j = 0; j < n_landmarks; j++) {
            printf("%d %d\n", j, assignment->assignment[j]);
        }
    }
}

__device__ void associate_landmarks_measurements(int i, float *particle, float *m, int *data_assoc_memory, float measurements[][2], int n_landmarks, int n_measurements, assignment *assignment, float *measurement_cov, float threshold) {
    dist_matrix matrix;
    matrix.matrix = m;

    matrix.n_landmarks = n_landmarks;
    matrix.n_measurements = n_measurements;

    compute_dist_matrix(particle, measurements, &matrix, measurement_cov);

    assign(i, &matrix, data_assoc_memory, assignment, threshold);
}

__device__ void update_landmark(float *particle, float measurements[][2], assignment *assignment, int n_measurements, float *measurement_cov)
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

        float *cov = get_cov(particle, i);

        float Q[4] = {
            cov[0] + measurement_cov[0],
            cov[1] + measurement_cov[1],
            cov[2] + measurement_cov[2],
            cov[3] + measurement_cov[3]
        };

        float K[4] = { 0, 0, 0, 0 };
        float Q_inv[4] = { 0, 0, 0, 0 };
        pinv(Q, Q_inv);
        matmul(cov, Q_inv, K);

        float K_residual[] = { 0, 0 };
        vecmul(K, residual, K_residual);
        mean[0] += K_residual[0];
        mean[1] += K_residual[1];


        float new_cov[] = { 1 - K[0], K[1], K[2], 1 - K[3] };
        matmul(new_cov, cov, new_cov);
        cov[0] = new_cov[0];
        cov[1] = new_cov[1];
        cov[2] = new_cov[2];
        cov[3] = new_cov[3];

        particle[3] *= pdf(measurements[j], measurement_predicted, Q);
    }
}

__device__ int get_max_landmarks_in_block(float *particles, int block_size, int i, int n_particles) {
    int max_landmarks = 0;

    for(int k = 0; k < block_size; k++) {
        float *particle = get_particle(particles, i*block_size + k);
        int n_landmarks = get_n_landmarks(particle);

        if(n_landmarks > max_landmarks) {
            max_landmarks = n_landmarks;
        }

        if((i*block_size + k) >= n_particles) {
            break;
        }
    }

    return max_landmarks;
}

__global__ void update(
    float *particles, int block_size, float measurements[][2], int n_particles, int n_measurements,
    float *measurement_cov, float threshold/*, int *scratchpad_memory, int size*/)
{
    // int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(n_measurements == 0) {
        return;
    }

    int blockId = blockIdx.x+ blockIdx.y * gridDim.x;
    int i = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int max_landmarks = get_max_landmarks_in_block(particles, block_size, i, n_particles);

    // int *scratchpad = scratchpad_memory + i*size;
    // int *assignment_memory = scratchpad;
    // int *data_assoc_memory = assignment_memory + (2 * max_landmarks + n_measurements);
    // float *matrix = (float *)(data_assoc_memory + (3 * max_landmarks * n_measurements));

    int scratchpad_size = (
            max_landmarks + n_measurements +
            (4 * max_landmarks * n_measurements)) * sizeof(int);
    int *scratchpad;
    int *assignment_memory;
    int *data_assoc_memory;
    float *matrix_memory;

    if(scratchpad_size > 0) {
        scratchpad = (int *)malloc(scratchpad_size);
        assignment_memory = scratchpad;
        data_assoc_memory = assignment_memory + (max_landmarks + n_measurements);
        matrix_memory = (float *)(data_assoc_memory + (3 * max_landmarks * n_measurements));
    }


    for(int k = 0; k < block_size; k++) {
        if((i*block_size + k) >= n_particles) {
            return;
        }
        
        float *particle = get_particle(particles, i*block_size + k);
        int n_landmarks = get_n_landmarks(particle);
    
        if(n_landmarks == 0) {
            add_measurements_as_landmarks(particle, measurements, n_measurements, measurement_cov);
            continue;
        }


        bool *assigned_measurements = (bool *)(assignment_memory);
        int *assignment_lm = (int *)(assignment_memory + n_measurements);

        for(int i = 0; i < n_measurements; i++) {
            assigned_measurements[i] = false;
        }
        
        for(int i = 0; i < n_landmarks; i++) {
            assignment_lm[i] = -1;
        }

        assignment assignmentx;
        assignmentx.assignment = assignment_lm;
        assignmentx.assigned_measurements = assigned_measurements;
    
        associate_landmarks_measurements(
            i, particle, matrix_memory, data_assoc_memory, measurements,
            n_landmarks, n_measurements, &assignmentx,
            measurement_cov, threshold
        );
    
        update_landmark(particle, measurements, &assignmentx, n_measurements, measurement_cov);
    
        add_unassigned_measurements_as_landmarks(particle, assignmentx.assigned_measurements, measurements, n_measurements, measurement_cov);
    
    }

    if(scratchpad_size > 0) {
        free(scratchpad);
    }
}