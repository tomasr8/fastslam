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

// __device__ float norm_squared(float *v) {
//     return v[0]*v[0] + v[1]*v[1];
// }

__device__ float vecnorm(float *v) {
    return sqrt(v[0]*v[0] + v[1]*v[1]);
}

__device__ bool in_sensor_range(float *position, float *landmark, float range, float fov) {
    float x = position[0];
    float y = position[1];
    float theta = position[2];
    float lx = landmark[0];
    float ly = landmark[1];

    float va[] = {lx - x, ly - y};
    float vb[] = {range * cos(theta), range * sin(theta)};

    if(vecnorm(va) > range) {
        return false;
    }

    float angle = acos(
        (va[0]*vb[0] + va[1]*vb[1])/(vecnorm(va)*vecnorm(vb))
    );

    if(angle <= (fov/2)) {
        return true;
    } else {
        return false;
    }
}

__device__ float* get_particle(float *particles, int i) {
    int max_landmarks = (int)particles[4];
    return (particles + (6 + 7*max_landmarks)*i);
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

__device__ float* get_landmark_prob(float *particle, int i)
{
    int max_landmarks = (int)particle[4];
    return (particle + 6 + 6*max_landmarks + i);
}

__device__ void increment_landmark_prob(float *particle, int i)
{
    int max_landmarks = (int)particle[4];
    float *prob = (particle + 6 + 6*max_landmarks + i);
    prob[0] += 1.0;
}

__device__ void decrement_landmark_prob(float *particle, int i)
{
    int max_landmarks = (int)particle[4];
    float *prob = (particle + 6 + 6*max_landmarks + i);
    prob[0] -= 1.0;
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
    float *new_prob = get_landmark_prob(particle, n_landmarks);

    new_mean[0] = mean[0];
    new_mean[1] = mean[1];

    new_cov[0] = cov[0];
    new_cov[1] = cov[1];
    new_cov[2] = cov[2];
    new_cov[3] = cov[3];

    new_prob[0] = 1.0;
}

__device__ void remove_landmark(float *particle, int i)
{
    int n_landmarks = (int)particle[5];

    for(int j = i + 1; j < n_landmarks; j++) {
        float *prob_a = get_landmark_prob(particle, j - 1);
        float *prob_b = prob_a + 1;

        prob_a[0] = prob_b[0];
    }
    
    for(int j = i + 1; j < n_landmarks; j++) {
        float *cov_a = get_cov(particle, j - 1);
        float *cov_b = cov_a + 4;

        cov_a[0] = cov_b[0];
        cov_a[1] = cov_b[1];
        cov_a[2] = cov_b[2];
        cov_a[3] = cov_b[3];
    }

    for(int j = i + 1; j < n_landmarks; j++) {
        float *mean_a = get_mean(particle, j - 1);
        float *mean_b = mean_a + 2;

        mean_a[0] = mean_b[0];
        mean_a[1] = mean_b[1];
    }

    particle[5] = (float)(n_landmarks - 1);
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

__device__ void update_landmark(float *particle, landmark_measurements *measurements, assignment *assignment, float range, float fov)
{
    float *measurement_cov = measurements->measurement_cov;

    float x = particle[0];
    float y = particle[1];
    int n_landmarks = get_n_landmarks(particle);

    for(int i = n_landmarks - 1; i >= 0; i--) {
        int j = assignment->assignment[i];

        if(j == -1) {
            if(in_sensor_range(particle, get_mean(particle, i), range, fov)) {
                decrement_landmark_prob(particle, i);
                float prob = get_landmark_prob(particle, i)[0];
                if(prob < 0) {
                    remove_landmark(particle, i);
                }
            }
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
        increment_landmark_prob(particle, i);
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
    float *particles, float *new_particles, int block_size, float measurements_array[][2], int n_particles, int n_measurements,
    float *measurement_cov, float threshold, float range, float fov)
{

    if(n_measurements == 0) {
        return;
    }

    int block_id = blockIdx.x+ blockIdx.y * gridDim.x;
    int thread_id = block_id * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    int max_landmarks = get_max_landmarks_in_block(particles, block_size, thread_id, n_particles);

    int scratchpad_size = (max_landmarks + n_measurements + (3 * max_landmarks * n_measurements) + n_particles) * sizeof(int);
    int *scratchpad;
    int *assignment_memory;
    int *data_assoc_memory;
    float *matrix_memory;
    float *cumsum;

    scratchpad = (int *)malloc(scratchpad_size);
    assignment_memory = scratchpad;
    data_assoc_memory = assignment_memory + (max_landmarks + n_measurements);
    matrix_memory = (float *)(data_assoc_memory + (2 * max_landmarks * n_measurements));
    cumsum = (float *)(matrix_memory + (max_landmarks * n_measurements));

    landmark_measurements measurements;
    measurements.n_measurements = n_measurements;
    measurements.measurement_cov = measurement_cov;
    measurements.measurements = measurements_array;

    for(int k = 0; k < block_size; k++) {
        int particle_id = thread_id*block_size + k;
        if(particle_id >= n_particles) {
            break;
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

        update_landmark(particle, &measurements, &assignmentx, range, fov);
    
        add_unassigned_measurements_as_landmarks(particle, assignmentx.assigned_measurements, &measurements);
    }



    // int particle_size = 6 + 7*((int)particles[4]);
    // int id_min = thread_id*block_size;
    // int id_max = MIN(thread_id*block_size + (block_size - 1), n_particles - 1);

    // float sum = 0;
    // for(int i = 0; i < n_particles; i++) {
    //     float w = get_particle(particles, i)[3];
    //     sum += w * w;
    // }

    // float neff = 1.0/sum;
    // if(thread_id == 0) {
    //     printf("neff: %f %f\n", neff, 0.6*n_particles);
    // }
    // if(neff > 0.6*n_particles) {
    //     for(int i = id_min; i <= id_max; i++) {
    //         float *new_particle = get_particle(new_particles, i);
    //         float *old_particle = get_particle(particles, i);

    //         memcpy(new_particle, old_particle, particle_size);

    //         // for(int k = 0; k < particle_size; k++) {
    //         //     new_particle[k] = old_particle[k];
    //         // }
    //     }
    //     free(scratchpad);
    //     return;
    // }

    // cumsum[0] = get_particle(particles, 0)[3];

    // for(int i = 1; i < n_particles; i++) {
    //     cumsum[i] = cumsum[i-1] + get_particle(particles, i)[3];
    // }

    // cumsum[n_particles-1] = 1.0;

    // int i = 0;
    // int j = 0;
    // while(i < n_particles) {
    //     if( ((i + random)/n_particles) < cumsum[j] ){

    //         if(i >= id_min && i <= id_max) {
    //             float *new_particle = get_particle(new_particles, i);
    //             float *old_particle = get_particle(particles, j);

    //             memcpy(new_particle, old_particle, particle_size);
    //             // for(int k = 0; k < particle_size; k++) {
    //             //     new_particle[k] = old_particle[k];
    //             // }

    //         }

    //         if(i > id_max) {
    //             break;
    //         }

    //         i += 1;
    //     } else {
    //         j += 1;
    //     }
    // }

    free(scratchpad);
}