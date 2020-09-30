import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import cv2 
import time
import pycuda.autoinit

matrices = np.random.randint(0, 5, size=(10000, 2, 2))
# print(matrices)
#nbtes determines the number of bytes for the numpy array a
img_gpu = cuda.mem_alloc(matrices.nbytes)
#Copies the memory from CPU to GPU
cuda.memcpy_htod(img_gpu, matrices)

mod = SourceModule("""
#include <stdio.h>
__global__ void pinv(float *A, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i >= n) {
        return;
    }

    float a = A[4*i + 0];
    float b = A[4*i + 1];
    float c = A[4*i + 2];
    float d = A[4*i + 3];

    float e = a*a + c*c;
    float f = a*b + c*d;
    float g = a*b + c*d;
    float h = b*b + d*d;

    float scalar = 1/(e*h - f*g);
    float e_i = scalar * h;
    float f_i = scalar * (-f);
    float g_i = scalar * (-g);
    float h_i = scalar * e;

    A[4*i + 0] = e_i*a + f_i*b;
    A[4*i + 1] = e_i*c + f_i*d;
    A[4*i + 2] = g_i*a + h_i*b;
    A[4*i + 3] = g_i*c + h_i*d;
}
""")
#Gives you the number of columns
n = matrices.shape[0]
func = mod.get_function("pinv")
start = time.time()
func(img_gpu, np.int32(n), block=(128, 1, 1))
img_ahe = np.empty_like(matrices)
cuda.memcpy_dtoh(img_ahe, img_gpu)

# print(img_ahe)
print(time.time() - start)