#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define M_PI 3.14159265359
#define MIN(a,b) (((a)<(b))?(a):(b))


typedef struct 
{
    float (*measurements)[2];
    int n_measurements;
    float *measurement_cov;
} landmark_measurements;

__device__ float mod_angle(float angle) {
    // angle = angle * 180.0 / M_PI;
    // angle = fmod(fmod(angle + 180.0, 360.0) + 360, 360.0) - 180;
    // angle = min + wrapMax(x - min, max - min)
    // angle = angle * M_PI / 180.0;
    // return angle;
    return atan2(sin(angle), cos(angle));
}

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

__device__ bool in_large_sensor_range(float *position, float *landmark, float range) {
    float x = position[0];
    float y = position[1];
    float lx = landmark[0];
    float ly = landmark[1];

    float dist_sq = (lx-x)*(lx-x) + (ly-y)*(ly-y);

    return dist_sq < range*range;
}

__device__ void to_coords(float *particle, float *in, float *out) {
    float x = particle[0];
    float y = particle[1];
    float theta = particle[2];

    float range = in[0];
    float bearing = in[1];

    out[0] = x + range * cos(bearing + theta);
    out[1] = y + range * sin(bearing + theta);
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

__device__ void matmul_jacobian(float *H, float *E, float *R, float *S)
{
    float a = H[0];
    float b = H[1];
    float c = H[2];
    float d = H[3];

    float Ht[] = {
        a, c,
        b, d
    };

    matmul(H, E, S);
    matmul(S, Ht, S);

    S[0] += R[0];
    S[1] += R[1];
    S[2] += R[2];
    S[3] += R[3];
}

__device__ void pinv(float *A, float *B)
{
    float a = A[0];
    float b = A[1];
    float c = A[2];
    float d = A[3];

    float scalar = 1/(a*d - b*c);

    B[0] = scalar * d;
    B[1] = scalar * (-b);
    B[2] = scalar * (-c);
    B[3] = scalar * a;
}

__device__ float pdf(float *x, float *mean, float* cov)
{
    float a = cov[0];
    float b = cov[1];

    printf("a^2-b^2(%f)\n", a*a - b*b);

    float logdet = log(a*a - b*b);

    printf("logdet(%f)\n", logdet);

    float root = sqrt(2.0)/2.0;
    float e = root * (1.0/sqrt(a-b));
    float f = root * (1.0/sqrt(a+b));

    printf("e(%f) f(%f)\n", e, f);

    float m = x[0] - mean[0];
    float n = x[1] - mean[1];

    float maha = 2*(m*m*e*e + n*n*f*f);
    printf("maha(%f)\n", maha);
    float log2pi = log(2 * M_PI);
    return exp(-0.5 * (2*log2pi + maha + logdet));
}

__device__ float pdf2(float *x, float *mean, float* cov)
{
    float cov_inv[] = {0, 0, 0, 0};
    pinv(cov, cov_inv);

    // printf("cov_inv[%f, %f, %f, %f]\n",
    //     cov_inv[0], cov_inv[1], cov_inv[2], cov_inv[3]
    // );

    float scalar = 1/(2*M_PI*sqrt(cov[0]*cov[3] - cov[1]*cov[2]));

    float m = x[0] - mean[0];
    float n = x[1] - mean[1];

    // float arg = m*m*(cov_inv[0] + cov_inv[2]) + n*n*(cov_inv[1] + cov_inv[3]);
    float arg = m*m*(cov_inv[0]) + n*n*(cov_inv[3]) + m*n*(cov_inv[1] + cov_inv[2]);

    // printf("scalar(%f) arg(%f)\n", scalar, arg);
    return scalar * exp(-0.5 * arg);
}

__device__ void add_measurement_as_landmark(float *particle, float *measurement, float *measurement_cov)
{
    float pos[] = { particle[0], particle[1] };
    float landmark[] = {0, 0};
    to_coords(particle, measurement, landmark);

    float q = (landmark[0] - pos[0])*(landmark[0] - pos[0]) + (landmark[1] - pos[1])*(landmark[1] - pos[1]);

    float H[] = {
        (landmark[0] - pos[0])/(2*sqrt(q)), (landmark[1] - pos[1])/(2*sqrt(q)),
        -(landmark[1] - pos[1])/q, (landmark[0] - pos[0])/q
    };

    // printf("m %f %f \n", measurement[0], measurement[1]);
    // printf("lm %f %f \n", landmark[0], landmark[1]);
    // printf("R %f %f %f %f \n", measurement_cov[0], measurement_cov[1], measurement_cov[2], measurement_cov[3]);
    // printf("H [%f, %f, %f, %f] \n", H[0], H[1], H[2], H[3]);

    pinv(H, H);
    // printf("Hinv %f %f %f %f \n", H[0], H[1], H[2], H[3]);

    float H_inv_t[] = {
        H[0], H[2],
        H[1], H[3]
    };

    float S[] = {
        0, 0, 0, 0
    };

    matmul(H, measurement_cov, S);
    // printf("S1 %f %f %f %f \n", S[0], S[1], S[2], S[3]);
    matmul(S, H_inv_t, S);
    // printf("S [%f, %f, %f, %f] \n", S[0], S[1], S[2], S[3]);

    // printf("pos[%f, %f, %f], m[%f, %f]\n", particle[0], particle[1], particle[2], measurement[0], measurement[1]);
    // printf("adding landmark [%f, %f] [%f, %f, %f, %f]\n", landmark[0], landmark[1], S[0], S[1], S[2], S[3]);
    // printf("=\n");

    add_landmark(particle, landmark, S);
}


__device__ void add_measurements_as_landmarks(float *particle, landmark_measurements *measurements)
{
    int n_measurements = measurements->n_measurements;
    float *measurement_cov = measurements->measurement_cov;

    for(int i = 0; i < n_measurements; i++) {
        add_measurement_as_landmark(particle, measurements->measurements[i], measurement_cov);
    }
}


__device__ float compute_dist(float *particle, int i, float *measurement, float *measurement_cov)
{
    float pos[] = { particle[0], particle[1] };
    float theta = particle[2];
    float *landmark_cov = get_cov(particle, i);
    float *landmark = get_mean(particle, i);

    // float coords[] = {0, 0};
    // to_coords(particle, measurement, coords);

    // printf("landmark[%f, %f], measurement[%f, %f], cov[%f, %f, %f, %f]\n",
    //     landmark[0], landmark[1], coords[0], coords[1], landmark_cov[0], landmark_cov[1], landmark_cov[2], landmark_cov[3]
    // );

    // float measurement_predicted[] = {
    //     landmark[0] - pos[0], landmark[1] - pos[1]
    // };

    // coords[0] -= pos[0];
    // coords[1] -= pos[1];


    float q = (landmark[0] - pos[0])*(landmark[0] - pos[0]) + (landmark[1] - pos[1])*(landmark[1] - pos[1]);

    float measurement_predicted[] = {
        sqrt(q), mod_angle(atan2(landmark[1] - pos[1], landmark[0] - pos[0]) - theta)
    };

    float H[] = {
        (landmark[0] - pos[0])/(2*sqrt(q)), (landmark[1] - pos[1])/(2*sqrt(q)),
        -(landmark[1] - pos[1])/q, (landmark[0] - pos[0])/q
    };

    float S[] = {
        0, 0, 0, 0
    };

    matmul_jacobian(H, landmark_cov, measurement_cov, S);

    // printf("S(%d) = [%f, %f, %f, %f]\n", i, S[0], S[1], S[2], S[3]);
    // printf("MP[%f, %f] M[%f, %f] S[%f, %f, %f, %f]\n",
    //     measurement_predicted[0], measurement_predicted[1], measurement[0], measurement[1],
    //     landmark_cov[0], landmark_cov[1], landmark_cov[2], landmark_cov[3]
    // );
    // printf("pdf(%f)\n", pdf2(landmark, coords, landmark_cov));
    // return pdf2(landmark, coords, landmark_cov);
    return pdf2(measurement_predicted, measurement, S);

}


__device__ void update_landmarks(int id, float *particle, landmark_measurements *measurements, int *in_range, int *n_matches, float range, float fov, float thresh)
{
    float *measurement_cov = measurements->measurement_cov;
    int n_measurements = measurements->n_measurements;

    int n_landmarks = get_n_landmarks(particle);

    int n_in_range = 0;
    for(int i = 0; i < n_landmarks; i++) {
        n_matches[i] = 0;
        float *mean = get_mean(particle, i);
        in_range[n_in_range] = i;
        n_in_range++;
        // if(in_large_sensor_range(particle, mean, range + 2)) {
        //     in_range[n_in_range] = i;
        //     n_in_range++;
        // }
    }

    // printf("START n_in_range(%d), n_landmarks(%d)\n", n_in_range, n_landmarks);
    for(int i = 0; i < n_measurements; i++) {
        float best = -1;
        int best_idx = -1;

        for(int j = 0; j < n_in_range; j++) {
            float dist = compute_dist(particle, in_range[j], measurements->measurements[i], measurement_cov);
            // printf("dist[%d, %d] = %f\n", j, i, dist);

            if(dist >= thresh && dist > best) {
                best = dist;
                best_idx = in_range[j];
            }
        }

        // if(id == 0) {
        //     printf("best dist: %f, thresh: %f\n", best, thresh);
        // }

        if(best_idx != -1) {
            n_matches[best_idx]++;
        }


        // printf("Measurement(%d) matched to (%d)\n", i, best_idx);

        if(best_idx != -1) {
            float *landmark = get_mean(particle, best_idx);
            float pos[] = { particle[0], particle[1] };
            float theta = particle[2];

            float q = (landmark[0] - pos[0])*(landmark[0] - pos[0]) + (landmark[1] - pos[1])*(landmark[1] - pos[1]);
            float measurement_predicted[] = {
                sqrt(q), mod_angle(atan2(landmark[1] - pos[1], landmark[0] - pos[0]) - theta)
            };

            float residual[2] = {
                measurements->measurements[i][0] - measurement_predicted[0],
                measurements->measurements[i][1] - measurement_predicted[1]
            };

            // printf("residual[%f, %f]\n", residual[0], residual[1]);

            float H[] = {
                (landmark[0] - pos[0])/(2*sqrt(q)), (landmark[1] - pos[1])/(2*sqrt(q)),
                -(landmark[1] - pos[1])/q, (landmark[0] - pos[0])/q
            };

            float Ht[] = {
                H[0], H[2],
                H[1], H[3]
            };

            float S[] = {
                0, 0, 0, 0
            };

            float *landmark_cov = get_cov(particle, best_idx);
        
            matmul_jacobian(H, landmark_cov, measurement_cov, S);
            float S_inv[] = {0, 0, 0, 0};
            pinv(S, S_inv);


            float Q[] = {0, 0, 0, 0};
            float K[] = { 0, 0, 0, 0 };
            matmul(landmark_cov, Ht, Q);
            matmul(Q, S_inv, K);

            float K_residual[] = { 0, 0 };
            vecmul(K, residual, K_residual);
            landmark[0] += K_residual[0];
            landmark[1] += K_residual[1];

            float KH[] = { 0, 0, 0, 0};
            matmul(K, H, KH);
            float new_cov[] = { 1 - KH[0], -KH[1], -KH[2], 1 - KH[3] };
            matmul(new_cov, landmark_cov, new_cov);
            landmark_cov[0] = new_cov[0];
            landmark_cov[1] = new_cov[1];
            landmark_cov[2] = new_cov[2];
            landmark_cov[3] = new_cov[3];

            // printf("new cov(%d) [%f, %f, %f, %f]\n", i,
            //     landmark_cov[0], landmark_cov[1], landmark_cov[2], landmark_cov[3]
            // );

            particle[3] *= pdf2(measurements->measurements[i], measurement_predicted, S);
            increment_landmark_prob(particle, best_idx);

            // float *landmark = get_mean(particle, best_idx);
            // float pos[] = { particle[0], particle[1] };
            // float theta = particle[2];

            // float q = (landmark[0] - pos[0])*(landmark[0] - pos[0]) + (landmark[1] - pos[1])*(landmark[1] - pos[1]);
            // float measurement_predicted[] = {
            //     sqrt(q), atan2(landmark[1] - pos[1], landmark[0] - pos[0]) - theta
            // };

            // float residual[2] = {
            //     measurements->measurements[i][0] - measurement_predicted[0],
            //     measurements->measurements[i][1] - measurement_predicted[1]
            // };

            // float H[] = {
            //     (landmark[0] - pos[0])/(2*sqrt(q)), (landmark[1] - pos[1])/(2*sqrt(q)),
            //     -(landmark[1] - pos[1])/q, (landmark[0] - pos[0])/q
            // };

            // float Ht[] = {
            //     H[0], H[2],
            //     H[1], H[3]
            // };

            // float S[] = {
            //     0, 0, 0, 0
            // };

            // float *landmark_cov = get_cov(particle, best_idx);
        
            // matmul_jacobian(H, landmark_cov, measurement_cov, S);
            // float S_inv[] = {0, 0, 0, 0};
            // pinv(S, S_inv);


            // float Q[] = {0, 0, 0, 0};
            // float K[] = { 0, 0, 0, 0 };
            // matmul(landmark_cov, H, Q);
            // matmul(Q, S_inv, K);

            // float K_residual[] = { 0, 0 };
            // vecmul(K, residual, K_residual);
            // landmark[0] += K_residual[0];
            // landmark[1] += K_residual[1];

            // float KH[] = { 0, 0, 0, 0};
            // matmul(K, Ht, KH);
            // float new_cov[] = { 1 - KH[0], -KH[1], -KH[2], 1 - KH[3] };
            // matmul(new_cov, landmark_cov, new_cov);
            // landmark_cov[0] = new_cov[0];
            // landmark_cov[1] = new_cov[1];
            // landmark_cov[2] = new_cov[2];
            // landmark_cov[3] = new_cov[3];

            // // printf("new cov(%d) [%f, %f, %f, %f]\n", i,
            // //     landmark_cov[0], landmark_cov[1], landmark_cov[2], landmark_cov[3]
            // // );

            // particle[3] *= pdf2(measurements->measurements[i], measurement_predicted, S);
            // increment_landmark_prob(particle, best_idx);
        } else {
            add_measurement_as_landmark(particle, measurements->measurements[i], measurement_cov);
        }
    }
    // for(int i = n_in_range - 1; i > 0; i--) {
    //     int idx = in_range[i];
    //     if(n_matches[idx] == 0) {
    //         decrement_landmark_prob(particle, idx);
    //         float prob = get_landmark_prob(particle, idx)[0];
    //         if(prob < 0) {
    //             remove_landmark(particle, idx);
    //         }
    //     } 
    // }

    // n_landmarks = get_n_landmarks(particle);
    // for(int i = 0; i < n_landmarks; i++) {
    //     float *mean = get_mean(particle, i);
    //     printf("landmark(%d) = [%f, %f]\n", i, mean[0], mean[1]);
    // }
    // for(int i = 0; i < n_landmarks; i++) {
    //     float *cov = get_cov(particle, i);
    //     printf("cov(%d) = [%f, %f, %f, %f]\n", i, cov[0], cov[1], cov[2], cov[3]);
    // }
}

__global__ void update(
    float *particles, int block_size, int *scratchpad_mem, int scratchpad_size, float measurements_array[][2], int n_particles, int n_measurements,
    float *measurement_cov, float threshold, float range, float fov, int max_landmarks)
{
    // int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(n_measurements == 0) {
        return;
    }

    // printf("n_measurements(%d)\n", n_measurements);

    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    int thread_id = block_id * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    // if(thread_id == 0) {
    //     for(int i = 0; i < n_measurements; i++) {
    //         printf("%d [%f %f]\n", i, measurements_array[i][0], measurements_array[i][1]);
    //     }
    // }

    int *scratchpad = scratchpad_mem + (2 * thread_id * max_landmarks);
    int *in_range = scratchpad;
    int *n_matches = in_range + max_landmarks;

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
            // n_landmarks = get_n_landmarks(particle);
            // for(int i = 0; i < n_landmarks; i++) {
            //     float *mean = get_mean(particle, i);
            //     printf("landmark(%d) = [%f, %f]\n", i, mean[0], mean[1]);
            // }
            // for(int i = 0; i < n_landmarks; i++) {
            //     float *cov = get_cov(particle, i);
            //     printf("cov(%d) = [%f, %f, %f, %f]\n", i, cov[0], cov[1], cov[2], cov[3]);
            // }
            continue;
        }

        update_landmarks(particle_id, particle, &measurements, in_range, n_matches, range, fov, threshold);

        // n_landmarks = get_n_landmarks(particle);
        // for(int i = 0; i < n_landmarks; i++) {
        //     float *mean = get_mean(particle, i);
        //     printf("landmark(%d) = [%f, %f]\n", i, mean[0], mean[1]);
        // }
        // for(int i = 0; i < n_landmarks; i++) {
        //     float *cov = get_cov(particle, i);
        //     printf("cov(%d) = [%f, %f, %f, %f]\n", i, cov[0], cov[1], cov[2], cov[3]);
        // }
    }
}