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
__global__ void matmul(float *A, float *B, float *C, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i >= n) {
        return;
    }

    float a = A[4*i + 0];
    float b = A[4*i + 1];
    float c = A[4*i + 2];
    float d = A[4*i + 3];

    float e = B[4*i + 0];;
    float f = B[4*i + 1];;
    float g = B[4*i + 2];;
    float h = B[4*i + 3];;

    C[4*i + 0] = a*e + b*g;
    C[4*i + 1] = a*f + b*h;
    C[4*i + 2] = c*e + d*g;
    C[4*i + 3] = c*f + d*h;
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