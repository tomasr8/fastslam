#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define PARTICLE_SIZE <<PARTICLE_SIZE>>

__device__ float* get_particle(float *particles, int i) {
    return (particles + PARTICLE_SIZE*i);
}

__device__ float* get_landmark_position(float *particle, int i)
{
    return (particle + 6 + 2*i);
}

/*
 * Finds particle with the maximum weight and uses its landmarks
 * to initialize kmeans.
 */
__global__ void initialize_centroids(
    float *particles, float *weights, int n_particles, float *centroids, int *n_centroids)
{
    float max_weight = 0;
    float best_index = 0;

    for(int i = 0; i < n_particles; i++) {
        if(weights[i] > max_weight) {
            max_weight = weights[i];
            best_index = i;
        }
    }

    float *particle = get_particle(particles, best_index);
    int n_landmarks = (int)particle[5]; 
    n_centroids[0] = n_landmarks;

    for(int i = 0; i < n_landmarks; i++) {
        float *landmark = get_landmark_position(particle, i);
        centroids[2*i] = landmark[0];
        centroids[2*i+1] = landmark[1];
    }
}


/*
 * Assigns landmarks to the closest centroids.
 */
__global__ void relabel(
    float *particles, float *labels, float *centroids, int block_size, int n_particles, int n_centroids)
{
    // int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int block_id = blockIdx.x+ blockIdx.y * gridDim.x;
    int thread_id = block_id * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    float *particle = get_particle(particles, thread_id);
    float *particle_labels = get_particle(labels, thread_id);
    float weight = particle[3];
    int n_landmarks = (int)particle[5];

    for(int j = 0; j < n_centroids; j++) {
        particle_labels[3*j] = 0.0;
        particle_labels[3*j+1] = 0.0;
        particle_labels[3*j+2] = 0.0;
    }

    // memset(particle_labels, 0, 4 * 3 * n_centroids);

    for(int j = 0; j < n_landmarks; j++) {
        float min_dist = 1e30;
        int best_centroid = 0;
        float *position = get_landmark_position(particle, j);

        for(int k = 0; k < n_centroids; k++) {
            float *centroid = centroids + 2*k;
            float dist =
                (position[0]-centroid[0])*(position[0]-centroid[0]) +
                (position[1]-centroid[1])*(position[1]-centroid[1]);

            if(dist < min_dist) {
                min_dist = dist;
                best_centroid = k;
            }
        }

        particle_labels[3*best_centroid] += weight * position[0];
        particle_labels[(3*best_centroid)+1] += weight * position[1];
        particle_labels[(3*best_centroid)+2] += weight;
    }
}


/*
 * Computes new centroids based on the landmark assignment.
 */
__global__ void compute_centroids(
    float *particles, float *labels, float *centroids, int n_particles, int n_centroids)
{
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    float sx = 0;
    float sy = 0;
    float denom = 0;

    for(int i = 0; i < n_particles; i++) {
        float *particle_labels = get_particle(labels, i);

        float weight = particle_labels[3*thread_id+2];

        sx += particle_labels[3*thread_id];
        sy += particle_labels[3*thread_id+1];
        denom += weight;
    }

    centroids[2*thread_id] = sx / denom;
    centroids[2*thread_id+1] = sy / denom;
}