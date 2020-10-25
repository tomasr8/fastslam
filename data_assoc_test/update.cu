#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define M_PI 3.14159265359
// particle

__device__ int min(int x, int y) { return (x<y)? x :y; }


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

__device__ void reverse(float arr[], int n)
{
    int temp;
    int start = 0;
    int end = n - 1;
    while (start < end)
    {
        temp = arr[start];   
        arr[start] = arr[end];
        arr[end] = temp;
        start++;
        end--;
    }   
} 

__device__ void insertionSort(float arr[], int n) 
{ 
    int i, key, j; 
    for (i = 1; i < n; i++) { 
        key = arr[i]; 
        j = i - 1; 
  
        /* Move elements of arr[0..i-1], that are 
          greater than key, to one position ahead 
          of their current position */
        while (j >= 0 && arr[j] > key) { 
            arr[j + 1] = arr[j]; 
            j = j - 1; 
        } 
        arr[j + 1] = key; 
    } 
} 


// __device__ int partition (float arr[], int low, int high) 
// { 
//     int pivot = arr[high];    // pivot 
//     int i = (low - 1);  // Index of smaller element 
  
//     for (int j = low; j <= high- 1; j++) 
//     { 
//         // If current element is smaller than the pivot 
//         if (arr[j] < pivot) 
//         { 
//             i++;    // increment index of smaller element 
//             swap(&arr[i], &arr[j]); 
//         } 
//     } 
//     swap(&arr[i + 1], &arr[high]); 
//     return (i + 1); 
// } 
  
// /* The main function that implements QuickSort 
//  arr[] --> Array to be sorted, 
//   low  --> Starting index, 
//   high  --> Ending index */
// __device__ void quickSort(float arr[], int low, int high) 
// { 
//     if (low < high) 
//     { 
//         /* pi is partitioning index, arr[p] is now 
//            at right place */
//         int pi = partition(arr, low, high); 
  
//         // Separately sort elements before 
//         // partition and after partition 
//         quickSort(arr, low, pi - 1); 
//         quickSort(arr, pi + 1, high); 
//     } 
// } 


__device__ int partition(float arr[], int l, int h)
{ 
    int x = arr[h]; 
    int i = (l - 1); 

    for (int j = l; j <= h - 1; j++) { 
        if (arr[j] > x) {
            i++; 
            swap(&arr[i], &arr[j]);
            // swap_idx(&lm[i], &lm[j]);
            // swap_idx(&me[i], &me[j]);
        } 
    } 
    swap(&arr[i + 1], &arr[h]);
    // swap_idx(&lm[i + 1], &lm[h]);
    // swap_idx(&me[i + 1], &me[h]);
    return (i + 1);
} 

__device__ void quickSort(float arr[], int l, int h) 
{ 
    int *stack = (int*)malloc((h-l+1) * sizeof(int));
    // int stack[1000];

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
        int p = partition(arr, l, h); 

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

__device__ void assign(float *particle, dist_matrix *matrix, assignment *assignment, float threshold) {
    int n_landmarks = matrix->n_landmarks;
    int n_measurements = matrix->n_measurements;

    int *stack = (int*)particle;

    // int *landmark_idx = (int *)malloc(n_landmarks * n_measurements * sizeof(int));
    // int *measurement_idx = (int *)malloc(n_landmarks * n_measurements * sizeof(int));

    // for(int i = 0; i < n_landmarks; i++) {
    //     for(int j = 0; j < n_measurements; j++) {
    //         landmark_idx[i * n_measurements + j] = i;
    //         measurement_idx[i * n_measurements + j] = j;
    //     }
    // }

    // =================================
    // =================================
    // int usable = 0;
    // for(int i = 0; i < n_landmarks * n_measurements; i++) {
    //     if(matrix->matrix[i] > threshold) {
    //         usable++;
    //     } 
    // }

    // float *matrix_copy = (float *)malloc(usable * sizeof(float));


    // =================================
    // =================================


    // float *matrix_copy = (float *)malloc(n_landmarks * n_measurements * sizeof(float));
    // for (int i = 0; i < n_landmarks * n_measurements; i++) {
    //     matrix_copy[i] = matrix->matrix[i]; 
    // }

    // quicksort(matrix->matrix, landmark_idx, measurement_idx, 0, (n_landmarks * n_measurements) - 1);
    // quicksort(matrix->matrix, 0, (n_landmarks * n_measurements) - 1);
    // mergeSort(matrix->matrix, (n_landmarks * n_measurements));
    // quickSort(matrix->matrix, 0, (n_landmarks * n_measurements) - 1);


    // quickSort(matrix->matrix, 0, (n_landmarks * n_measurements) - 1, stack);

    // reverse(matrix->matrix, (n_landmarks * n_measurements) - 1);
    // quickSort(matrix->matrix, 0, (n_landmarks * n_measurements) - 1, stack);

    // reverse(matrix->matrix, (n_landmarks * n_measurements) - 1);
    // quickSort(matrix->matrix, 0, (n_landmarks * n_measurements) - 1, stack);

    // reverse(matrix->matrix, (n_landmarks * n_measurements) - 1);
    // quickSort(matrix->matrix, 0, (n_landmarks * n_measurements) - 1, stack);


    insertionSort(matrix->matrix, n_landmarks * n_measurements);

    for(int i = 0; i < 32; i++) {
        reverse(matrix->matrix, (n_landmarks * n_measurements) - 1);
        insertionSort(matrix->matrix, n_landmarks * n_measurements);
        // quickSort(matrix->matrix, 0, (n_landmarks * n_measurements) - 1);
    }

    // free(matrix_copy);

    int assigned_total = 0;
    // float cost = 0;

    // for(int i = 0; i < n_landmarks * n_measurements; i++) {
    //     int a = landmark_idx[i];
    //     int b = measurement_idx[i];

    //     if(assignment->assigned_landmarks[a] || assignment->assigned_measurements[b]){
    //         continue;
    //     }
        
    //     if(matrix->matrix[a * n_measurements + b] > threshold){
    //         assignment->assignment[a] = b;
    //         assignment->assigned_landmarks[a] = true;
    //         assignment->assigned_measurements[b] = true;
    //         assigned_total += 1;
    //         // cost += matrix->matrix[a * n_measurements + b];
    //     }

    //     if(assigned_total == n_landmarks) {
    //         break;
    //     }
    // }

    // printf("Cost: %f\n", cost);

    // free(landmark_idx);
    // free(measurement_idx);
}

__device__ void associate_landmarks_measurements(float *particle, float measurements[][2], int n_landmarks, int n_measurements, assignment *assignment, float *measurement_cov, float threshold) {
    if(n_landmarks > 0 && n_measurements > 0) {
        dist_matrix *matrix = (dist_matrix *)malloc(sizeof(dist_matrix));
        matrix->matrix = (float *)malloc(n_landmarks * n_measurements * sizeof(float));;
        matrix->n_landmarks = n_landmarks;
        matrix->n_measurements = n_measurements;

        compute_dist_matrix(particle, measurements, matrix, measurement_cov);

        assign(particle, matrix, assignment, threshold);

        free(matrix->matrix);
        free(matrix);
    }
}

__global__ void update(float *particles, float measurements[][2], int n_particles, int n_measurements, float *measurement_cov, float threshold)
{
    int blockId = blockIdx.x+ blockIdx.y * gridDim.x;
    int i = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if(i >= n_particles) {
        return;
    }

    float *particle = get_particle(particles, i);
    int n_landmarks = get_n_landmarks(particle);

    bool *assigned_landmarks = (bool *)malloc(n_landmarks * sizeof(bool));
    bool *assigned_measurements = (bool *)malloc(n_measurements * sizeof(bool));

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

    assignment *assignmentx = (assignment*)malloc(sizeof(assignment));
    assignmentx->assignment = assignment_lm;
    assignmentx->assigned_landmarks = assigned_landmarks;
    assignmentx->assigned_measurements = assigned_measurements;

    associate_landmarks_measurements(
        particle, measurements,
        n_landmarks, n_measurements, assignmentx,
        measurement_cov, threshold
    );

    free(assignmentx->assigned_landmarks);
    free(assignmentx->assigned_measurements);
    free(assignmentx->assignment);
    free(assignmentx);
}