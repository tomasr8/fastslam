#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define M_PI 3.14159265359
#define MIN(a,b) (((a)<(b))?(a):(b))

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
    float (*measurements)[2];
    int n_measurements;
    float *measurement_cov;
} landmark_measurements;

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

__device__ void add_unassigned_measurements_as_landmarks(float *particle, bool *assigned_measurements, landmark_measurements *measurements)
{
    int n_measurements = measurements->n_measurements;
    float *measurement_cov = measurements->measurement_cov;

    for(int i = 0; i < n_measurements; i++) {
        if(!assigned_measurements[i]) {
            float x = particle[0];
            float y = particle[1];
            float measurement[] = {
                x + measurements->measurements[i][0],
                y + measurements->measurements[i][1]
            };

            add_landmark(particle, measurement, measurement_cov);
        }
    }
}

__device__ void add_measurements_as_landmarks(float *particle, landmark_measurements *measurements)
{
    int n_measurements = measurements->n_measurements;
    float *measurement_cov = measurements->measurement_cov;

    for(int i = 0; i < n_measurements; i++) {
        float x = particle[0];
        float y = particle[1];
        float measurement[] = {
            x + measurements->measurements[i][0],
            y + measurements->measurements[i][1]
        };

        add_landmark(particle, measurement, measurement_cov);
    }
}

__device__ void insertion_sort(float arr[], int lm[], int me[], int n) 
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

__device__ void vecmul(float *A, float *u, float *v)
{
    float a = A[0];
    float b = A[1];
    float c = A[2];
    float d = A[3];

    float e = u[0];
    float f = v[1];

    v[0] = a*e + b*f;
    v[1] = c*e + d*f;
}

__device__ void matmul(float *A, float *B, float *C)
{
    float a = A[0];
    float b = A[1];
    float c = A[2];
    float d = A[3];

    float e = B[0];
    float f = B[1];
    float g = B[2];
    float h = B[3];

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

__device__ void compute_dist_matrix(float *particle, landmark_measurements *measurements, dist_matrix *matrix)
{
    float *measurement_cov = measurements->measurement_cov;
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

            matrix->matrix[i * matrix->n_measurements + j] = pdf(measurement_predicted, measurements->measurements[j], cov);
        }
    }
}

__device__ void assign(dist_matrix *matrix, int *data_assoc_memory, assignment *assignment, float threshold) {
    int n_landmarks = matrix->n_landmarks;
    int n_measurements = matrix->n_measurements;

    int *landmark_idx = data_assoc_memory;
    int *measurement_idx = landmark_idx + (n_landmarks * n_measurements);

    int k = 0;
    for(int i = 0; i < n_landmarks; i++) {
        for(int j = 0; j < n_measurements; j++) {
            // only take values > threshold
            if(matrix->matrix[i * n_measurements + j] > threshold) {
                landmark_idx[k] = i;
                measurement_idx[k] = j;
                matrix->matrix[k] = matrix->matrix[i * n_measurements + j];
                k++;
            }
        }
    }

    insertion_sort(matrix->matrix, landmark_idx, measurement_idx, k);

    int iterations = MIN(n_landmarks, k);
    for(int i = 0; i < iterations; i++) {
        int a = landmark_idx[i];
        int b = measurement_idx[i];

        if(assignment->assignment[a] != -1){
            continue;
        }

        assignment->assignment[a] = b;
        assignment->assigned_measurements[b] = true;
    }
}

__device__ void associate_landmarks_measurements(float *particle, float *m, int *data_assoc_memory, landmark_measurements *measurements, int n_landmarks, assignment *assignment, float threshold) {
    int n_measurements = measurements->n_measurements;
    
    dist_matrix matrix;
    matrix.matrix = m;
    matrix.n_landmarks = n_landmarks;
    matrix.n_measurements = n_measurements;

    compute_dist_matrix(particle, measurements, &matrix);

    assign(&matrix, data_assoc_memory, assignment, threshold);
}

__device__ void update_landmark(float *particle, landmark_measurements *measurements, assignment *assignment)
{
    float *measurement_cov = measurements->measurement_cov;

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
        float residual[2] = {
            measurements->measurements[j][0] - measurement_predicted[0],
            measurements->measurements[j][1] - measurement_predicted[1]
        };

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

        particle[3] *= pdf(measurements->measurements[j], measurement_predicted, Q);
    }
}

__device__ int get_max_landmarks_in_block(float *particles, int block_size, int thread_id, int n_particles) {
    int max_landmarks = 0;

    for(int k = 0; k < block_size; k++) {
        int particle_id = thread_id*block_size + k;
        if(particle_id >= n_particles) {
            break;
        }

        float *particle = get_particle(particles, particle_id);
        int n_landmarks = get_n_landmarks(particle);

        if(n_landmarks > max_landmarks) {
            max_landmarks = n_landmarks;
        }
    }

    return max_landmarks;
}

__global__ void update(
    float *particles, int block_size, float measurements_array[][2], int n_particles, int n_measurements,
    float *measurement_cov, float threshold/*, int *scratchpad_memory, int size*/)
{
    // int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(n_measurements == 0) {
        return;
    }

    int block_id = blockIdx.x+ blockIdx.y * gridDim.x;
    int thread_id = block_id * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int max_landmarks = get_max_landmarks_in_block(particles, block_size, thread_id, n_particles);

    int scratchpad_size = (
            max_landmarks + n_measurements +
            (3 * max_landmarks * n_measurements)) * sizeof(int);
    int *scratchpad;
    int *assignment_memory;
    int *data_assoc_memory;
    float *matrix_memory;

    if(scratchpad_size > 0) {
        scratchpad = (int *)malloc(scratchpad_size);
        assignment_memory = scratchpad;
        data_assoc_memory = assignment_memory + (max_landmarks + n_measurements);
        matrix_memory = (float *)(data_assoc_memory + (2 * max_landmarks * n_measurements));
    }

    landmark_measurements measurements;
    measurements.n_measurements = n_measurements;
    measurements.measurement_cov = measurement_cov;
    measurements.measurements = measurements_array;

    for(int k = 0; k < block_size; k++) {
        int particle_id = thread_id*block_size + k;
        if(particle_id >= n_particles) {
            return;
        }
        
        float *particle = get_particle(particles, particle_id);
        int n_landmarks = get_n_landmarks(particle);
    
        if(n_landmarks == 0) {
            add_measurements_as_landmarks(particle, &measurements);
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
            particle, matrix_memory, data_assoc_memory, &measurements,
            n_landmarks, &assignmentx,
            threshold
        );

        update_landmark(particle, &measurements, &assignmentx);
    
        add_unassigned_measurements_as_landmarks(particle, assignmentx.assigned_measurements, &measurements);
    }

    if(scratchpad_size > 0) {
        free(scratchpad);
    }
}