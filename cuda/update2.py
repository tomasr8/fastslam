import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

with open("cuda/update2.cu", "r") as f:
    source = f.read()

with open("cuda/resample.cu", "r") as f:
    resample = f.read()

with open("cuda/predict.cu", "r") as f:
    predict = f.read()

with open("cuda/mean.cu", "r") as f:
    mean = f.read()

cuda_update = SourceModule(source)
cuda_resample = SourceModule(resample)
cuda_predict = SourceModule(predict, no_extern_c=True)
cuda_mean = SourceModule(mean)