import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import cv2 
import time
import pycuda.autoinit

A = np.random.randint(0, 5, size=(10000, 2, 2))
B = np.random.randint(0, 5, size=(10000, 2, 2))
C = np.float32(size=(10000, 2, 2))

# print(matrices)
#nbtes determines the number of bytes for the numpy array a
cuda_A = cuda.mem_alloc(A.nbytes)
cuda_B = cuda.mem_alloc(B.nbytes)
cuda_C = cuda.mem_alloc(C.nbytes)

#Copies the memory from CPU to GPU
cuda.memcpy_htod(cuda_A, A)
cuda.memcpy_htod(cuda_B, B)
cuda.memcpy_htod(cuda_C, C)


mod = SourceModule("""
#include <stdio.h>

#define N_LANDMARKS 100

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

    float root = sqrt(2)/2.0;
    float e = root * (1/sqrt(a-b));
    float f = root * (1/sqrt(a+b));

    float m = x[0] - mean[0];
    float n = x[1] - mean[1];

    float maha = 2*(m*m*e*e + n*n*f*f);
    float log2pi = log(2 * math.pi);
    return exp(-0.5 * (2*log2pi + maha + logdet));
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

__device__ float* get_particle(float *particles, int i)
{
    return (particles + 5*i + 6*N_LANDMARKS*i);
}

__device__ float* get_mean(float *particle, int i)
{
    return (particle + 5 + 2*i);
}

__device__ float* get_covariance(float *particle, int i)
{
    return (particle + 5 + 2*N_LANDMARKS*i);
}

__device__ void update_landmark(float *particle, float **z_real, int *assignment, float *sigma, int n)
{

    float x = particle[0];
    float y = particle[1];

    for(int i = 0; i < n; i++) {
        int j = assignment[i];

        float *mean = get_mean(particle, i);
        float mean_x = mean[0];
        float mean_y = mean[1];

        float z_predicted[2] = { mean_x - x, mean_y - y };
        float residual[2] = { z_real[j][0] - z_predicted[0], z_real[j][1] - z_predicted[1] };

        float *cov = get_covariance(particle, i);

        float Q[4] = { cov[0] + sigma[0], cov[1], cov[2], cov[3] + sigma[1] };
        float K[4] = { 0, 0, 0, 0 };
        float Q_inv[4] = { 0, 0, 0, 0};
        pinv(Q, Q_inv);
        matmul(cov, Q_inv, K);

        particle[3] *= pdf(z_real[j], z_predicted, Q);
    }
}
""")
#Gives you the number of columns
n = A.shape[0]
func = mod.get_function("matmul")
start = time.time()
func(cuda_A, cuda_B, cuda_C, np.int32(n), block=(128, 1, 1))
out = np.empty_like(C)
cuda.memcpy_dtoh(out, cuda_C)

# print(img_ahe)
print(time.time() - start)