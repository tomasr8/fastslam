import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import cv2 
import time
import pycuda.autoinit

covs = np.random.randint(0, 5, size=(10000, 2))
# print(matrices)
#nbtes determines the number of bytes for the numpy array a
img_gpu = cuda.mem_alloc(covs.nbytes)
#Copies the memory from CPU to GPU
cuda.memcpy_htod(img_gpu, covs)

mod = SourceModule("""
#include <stdio.h>
#include <math.h>
__global__ void pdf(float *A, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i >= n) {
        return;
    }

    float a = A[2*i + 0];
    float b = A[2*i + 1];

    float logdet = log(a*a - b*b);

    float root = sqrt(2)/2.0;
    float e = root * (1/sqrt(a-b));
    float f = root * (1/sqrt(a+b));

    float m = 


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
n = covs.shape[0]
func = mod.get_function("pdf")
start = time.time()
func(img_gpu, np.int32(n), block=(128, 1, 1))
img_ahe = np.empty_like(covs)
cuda.memcpy_dtoh(img_ahe, img_gpu)

# print(img_ahe)
print(time.time() - start)